"""LaTeX layout and compilation for the single project report.

Generated TeX, image copies and compiler files live in a temporary directory.
Requires XeLaTeX locally or in Ubuntu WSL; no packages are downloaded.
"""
from __future__ import annotations

from datetime import datetime
from html import unescape
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


SYMBOLS = {
    'α': r'\ensuremath{\alpha}', 'β': r'\ensuremath{\beta}',
    'γ': r'\ensuremath{\gamma}', 'λ': r'\ensuremath{\lambda}',
    'σ': r'\ensuremath{\sigma}', 'Δ': r'\ensuremath{\Delta}',
    'ŷ': r'\ensuremath{\hat{y}}', '²': r'\textsuperscript{2}',
    '±': r'\ensuremath{\pm}', '×': r'\ensuremath{\times}',
    '·': r'\ensuremath{\cdot}', '↑': r'\ensuremath{\uparrow}',
    '↓': r'\ensuremath{\downarrow}', '≥': r'\ensuremath{\geq}',
    '°': r'\ensuremath{^\circ}',
}
ESCAPES = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
           '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
           '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def tex_text(value):
    return ''.join(SYMBOLS.get(c, ESCAPES.get(c, c)) for c in str(value))


class Inline(HTMLParser):
    """Translate the report's small HTML vocabulary, escaping ordinary text."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.math = [], False

    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            self.parts.append(r'\newline{}')
        elif tag == 'b':
            self.parts.append(r'\textbf{')
        elif tag == 'link':
            self.parts.append(r'\href{' + tex_text(dict(attrs)['href']) + '}{')
        elif tag == 'math':
            self.math = True
            self.parts.append(r'\(')
        else:
            raise ValueError('Unsupported report markup: ' + tag)

    def handle_endtag(self, tag):
        if tag in ('b', 'link'):
            self.parts.append('}')
        elif tag == 'math':
            self.parts.append(r'\)')
            self.math = False

    def handle_data(self, data):
        if self.math:
            greek = {'α': r'\alpha ', 'β': r'\beta ', 'γ': r'\gamma ', 'λ': r'\lambda ', 'σ': r'\sigma '}
            self.parts.append(''.join(greek.get(c, c) for c in data))
        elif data.startswith(('https://', 'http://')) and not any(c.isspace() for c in data):
            self.parts.append(r'\nolinkurl{' + data + '}')
        else:
            self.parts.append(tex_text(data))


def inline(value):
    parser = Inline()
    parser.feed(str(value))
    parser.close()
    return ''.join(parser.parts)


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[polish]{babel}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}[Scale=MatchLowercase]
\usepackage{microtype}
\usepackage[margin=20mm,top=22mm,bottom=21mm,headheight=23pt,headsep=6mm]{geometry}
\usepackage[table]{xcolor}
\usepackage{graphicx,booktabs,longtable,array,amsmath}
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{caption,float,pdflscape,fancyhdr,titlesec,xurl,lastpage}
\usepackage{tcolorbox}
\usepackage[unicode,colorlinks=true,linkcolor=navy,urlcolor=teal,citecolor=teal,
  bookmarksnumbered=true,pdfencoding=auto]{hyperref}
\definecolor{navy}{HTML}{17324D}
\definecolor{teal}{HTML}{087F8C}
\definecolor{muted}{HTML}{526171}
\definecolor{paper}{HTML}{EDF5F6}
\definecolor{stripe}{HTML}{F5F7F9}
\hypersetup{pdftitle={SIGK 2 - Rekonstrukcja zdjęć przy słabym oświetleniu},
  pdfauthor={__AUTHORS__},pdfsubject={Rekonstrukcja obrazów, eksperymenty U-Net, NAFNet i MIRNet}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{3em}
\linespread{1.04}
\raggedbottom
\widowpenalty=10000
\clubpenalty=10000
\setcounter{tocdepth}{2}
\makeatletter
\renewcommand{\l@subsection}{\@dottedtocline{2}{1.5em}{3.1em}}
\makeatother
\titleformat{\section}{\Large\sffamily\bfseries\color{navy}}{\thesection}{.65em}{}
\titleformat{\subsection}{\large\sffamily\bfseries\color{navy}}{\thesubsection}{.65em}{}
\titlespacing*{\section}{0pt}{0pt}{15pt}
\titlespacing*{\subsection}{0pt}{12pt}{7pt}
\captionsetup{font=small,labelfont={sf,bf},textfont={color=muted},skip=7pt}
\captionsetup[table]{position=top,justification=raggedright,singlelinecheck=false}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.26}
\setlength{\LTpre}{5pt}
\setlength{\LTpost}{9pt}
\setlength{\LTcapwidth}{\textwidth}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\sffamily\color{muted}SIGK / Projekt 2}
\fancyhead[R]{\footnotesize\sffamily\color{muted}Rekonstrukcja zdjęć}
\fancyfoot[L]{\footnotesize\sffamily\color{muted}Low light image enhancement}
\fancyfoot[R]{\footnotesize\sffamily\color{muted}\thepage}
\renewcommand{\headrulewidth}{.35pt}
\renewcommand{\footrulewidth}{0pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{teal}\leaders\hrule height\headrulewidth\hfill}}
\begin{document}
"""


class LatexReport:
    def __init__(self, authors):
        self.authors, self.parts, self.assets = authors, [], []
        self.current_heading = 'Zakres projektu'
        self.pending_caption = None
        self.landscape = False

    def cover(self, complete):
        authors = r'\\[3pt]'.join(tex_text(n.strip()) for n in self.authors.split(',') if n.strip())
        status = 'Zakończone eksperymenty' if complete else 'Wersja metodologiczna przed obliczeniami'
        self.parts.append(r'\begin{titlepage}\thispagestyle{empty}')
        self.parts.append(r'{\sffamily\small\color{teal}\bfseries SIGK / PROJEKT 2\par}\vspace{22mm}')
        self.parts.append(r'{\sffamily\bfseries\color{navy}\fontsize{34}{40}\selectfont Rekonstrukcja zdjęć\\przy słabym\\oświetleniu\par}')
        self.parts.append(r'\vspace{7mm}{\sffamily\Large\color{muted}Low light image enhancement\par}')
        self.parts.append(r'\vspace{11mm}{\color{teal}\rule{32mm}{1.4pt}}\par\vspace{8mm}')
        self.parts.append(r'{\sffamily\small\color{muted}AUTORZY\par}\vspace{2mm}{\large ' + authors + r'\par}')
        self.parts.append(r'\vspace{8mm}{\sffamily\small\color{muted}SPRAWOZDANIE Z PROJEKTU\par}'
                          + r'{\small ' + tex_text(datetime.now().strftime('%d.%m.%Y')) + r'\par}\vfill')
        self.parts.append(r'{\sffamily\small\color{teal}\bfseries ' + tex_text(status) + r'\par}')

    def start_body(self):
        self.parts.append(r'\vspace{5mm}{\color{teal}\rule{\linewidth}{.5pt}}\end{titlepage}')
        self.parts.append(r'\pagenumbering{roman}\tableofcontents\clearpage\pagenumbering{arabic}')

    def paragraph(self, text, style='BodyText'):
        if style == 'SmallText' and re.match(r'^Tabela \d+\.', text):
            self.pending_caption = re.sub(r'^Tabela \d+\.\s*', '', text)
            return
        rendered = inline(text)
        if style == 'Heading2':
            self.parts.append(r'\subsection*{' + rendered + '}')
        elif style == 'SmallText':
            self.parts.append(r'{\small\color{muted} ' + rendered + r'\par}')
        elif style == 'CodeText':
            self.parts.append(r'\begin{tcolorbox}[colback=paper,colframe=teal!30,boxrule=.4pt,arc=1mm,'
                              r'left=2mm,right=2mm,top=1.5mm,bottom=1.5mm]'
                              r'\ttfamily\footnotesize\raggedright ' + rendered + r'\end{tcolorbox}')
        else:
            self.parts.append(rendered + r'\par')

    def heading(self, text):
        self.current_heading = unescape(re.sub(r'^\d+\.\s*', '', text))
        rendered = inline(self.current_heading)
        if re.match(r'^\d+\. ', text):
            self.parts.append(r'\section{' + rendered + '}')
        elif text == 'Ograniczenia i źródła':
            self.parts.append(r'\section*{' + rendered + r'}\addcontentsline{toc}{section}{' + rendered + '}')
        elif ' / ' in text:
            if self.landscape:
                self.parts.append(r'{\large\sffamily\bfseries\color{navy} ' + rendered + r'}\hfill{\small\sffamily\color{muted}\thepage}\par\vspace{5pt}')
            else:
                self.parts.append(r'\subsection*{' + rendered + '}')
        else:
            self.parts.append(r'\subsection{' + rendered + '}')

    def page_break(self, landscape=False):
        if self.landscape:
            self.parts.append(r'\end{landscape}')
        self.parts.append(r'\clearpage')
        self.landscape = landscape
        if landscape:
            self.parts.append(r'\begin{landscape}\thispagestyle{empty}')

    def table(self, rows, widths=None, small=False):
        fractions = [w/sum(widths) for w in widths] if widths else [1/len(rows[0])]*len(rows[0])
        columns = ''.join(r'>{\raggedright\arraybackslash}p{\dimexpr ' + f'{w:.6f}'
                          + r'\linewidth-2\tabcolsep\relax}' for w in fractions)
        title = self.pending_caption or self.current_heading
        self.pending_caption = None
        size = r'\fontsize{8.2}{10.5}\selectfont' if small else r'\fontsize{9}{11.5}\selectfont'
        self.parts.append(r'\begingroup' + size + r'\rowcolors{2}{stripe}{white}')
        self.parts.append(r'\begin{longtable}{' + columns + '}')
        self.parts.append(r'\caption{' + inline(title) + r'}\\\toprule')
        header = ' & '.join(r'{\sffamily\bfseries\color{navy} ' + inline(str(c)) + '}' for c in rows[0])
        self.parts.append(header + r'\\\midrule\endfirsthead')
        self.parts.append(r'\multicolumn{' + str(len(rows[0])) + r'}{l}{\small\itshape Ciąg dalszy tabeli \thetable}\\\toprule')
        self.parts.append(header + r'\\\midrule\endhead\bottomrule\endfoot')
        for row in rows[1:]:
            self.parts.append(' & '.join(inline(str(c)) for c in row) + r'\\')
        self.parts.append(r'\end{longtable}\endgroup')

    def figure(self, path, caption, maxheight=230):
        path = Path(path).resolve()
        filename = f'figure-{len(self.assets)+1:02d}{path.suffix.lower()}'
        self.assets.append((path, filename))
        caption = re.sub(r'^Rys\.\s*\d+\.\s*', '', caption)
        height = min(maxheight, 270) if self.landscape else maxheight
        self.parts.append(r'\begin{figure}[H]\centering\includegraphics[width=\linewidth,height='
                          + f'{height}bp' + r',keepaspectratio]{' + filename + r'}\caption{' + inline(caption) + r'}\end{figure}')

    def compile(self, output, distro='Ubuntu', keep_build=None):
        if self.landscape:
            self.parts.append(r'\end{landscape}')
        source = PREAMBLE.replace('__AUTHORS__', tex_text(self.authors)) + '\n'.join(self.parts) + '\n' + r'\end{document}'
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_parent = Path(__file__).resolve().parent/'tmp'
        temp_parent.mkdir(exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix='latex_', dir=temp_parent) as folder:
                build = Path(folder)
                (build/'report.tex').write_text(source, encoding='utf-8')
                for path, filename in self.assets:
                    shutil.copyfile(path, build/filename)
                if shutil.which('xelatex'):
                    command = ['xelatex']
                    cwd = build
                elif os.name == 'nt' and shutil.which('wsl'):
                    linux_path = '/mnt/' + build.drive[0].lower() + build.as_posix()[2:]
                    command = ['wsl', '-d', distro, '--cd', linux_path, '--', 'xelatex']
                    cwd = None
                else:
                    raise RuntimeError('Brak XeLaTeX. Zainstaluj TeX Live lokalnie albo w Ubuntu WSL.')
                for _ in range(3):
                    result = subprocess.run(command + ['-interaction=nonstopmode', '-halt-on-error', '-file-line-error', 'report.tex'],
                                            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                    log = result.stdout.decode('utf-8', errors='replace')
                    if result.returncode:
                        if keep_build:
                            shutil.copytree(build, keep_build, dirs_exist_ok=True)
                        raise RuntimeError('Kompilacja LaTeX nie powiodla sie:\n' + log[-6000:])
                compiler_log = (build/'report.log').read_text(encoding='utf-8', errors='replace')
                if keep_build:
                    shutil.copytree(build, keep_build, dirs_exist_ok=True)
                warnings = [line for line in compiler_log.splitlines() if 'Overfull' in line or 'undefined' in line or 'Missing character' in line]
                serious = [line for line in warnings if 'undefined' in line or 'Missing character' in line
                           or (re.search(r'Overfull .*?\(([\d.]+)pt', line)
                               and float(re.search(r'Overfull .*?\(([\d.]+)pt', line)[1]) > 1)]
                if serious:
                    raise RuntimeError('Niepoprawny sklad LaTeX:\n' + '\n'.join(serious))
                if warnings:
                    print('LaTeX: uwagi do sprawdzenia:\n' + '\n'.join(warnings))
                temporary_output = output.with_suffix('.pdf.tmp')
                shutil.copyfile(build/'report.pdf', temporary_output)
                temporary_output.replace(output)
        finally:
            try:
                temp_parent.rmdir()
            except OSError:
                pass
