"""Build the Elsevier (Ultramicroscopy) version from paper/main.tex.

Author/affiliation lines are placeholders; fill them in before submission.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
s = (PAPER / "main.tex").read_text()

t0 = s.index(r"\title{") + len(r"\title{")
t1 = s.index("}\n" + chr(92) + "author", t0)
title = " ".join(s[t0:t1].replace(chr(92) * 2, " ").replace(r"\bfseries", "").split())

a0 = s.index(r"\begin{abstract}") + len(r"\begin{abstract}")
a1 = s.index(r"\end{abstract}", a0)
abstract = s[a0:a1].strip()
body = s[s.index(r"\section{Introduction}"):]

pre = r"""\documentclass[preprint,12pt]{elsarticle}
\usepackage{newtxtext,newtxmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{caption}
\usepackage{float}
\usepackage{lineno}
\captionsetup{font=small,labelfont=bf}
\graphicspath{{../figures/}}
\input{numbers}
\input{real_numbers}
\input{fmd_numbers}
\setlength{\emergencystretch}{2em}
\journal{Ultramicroscopy}
\begin{document}
\begin{frontmatter}
\title{%s}
%% TODO: replace the author/affiliation placeholders before submission.
\author[1]{[Author One]\corref{cor1}}
\ead{[author.one@example.com]}
\author[2]{[Author Two]}
\ead{[author.two@example.com]}
\affiliation[1]{organization={[Department, Institution]}, city={[City]}, country={[Country]}}
\affiliation[2]{organization={[Department, Institution]}, city={[City]}, country={[Country]}}
\cortext[cor1]{Corresponding author}
\begin{abstract}
%s
\end{abstract}
\begin{keyword}
fluorescence microscopy \sep deconvolution \sep low-dose imaging \sep deep learning \sep uncertainty
\end{keyword}
\end{frontmatter}
\linenumbers

""" % (title, abstract)
decl = r"""
\section*{CRediT authorship contribution statement}
[To be completed by the authors.]

\section*{Declaration of competing interest}
[To be completed by the authors.]

\section*{Funding}
[To be completed by the authors.]

\section*{Declaration of generative AI and AI-assisted technologies}
[To be completed by the authors.]

"""
body = body.replace(r"\begin{table}[t]", r"\begin{table}[H]")
body = body.replace(r"\section*{Data and code availability}",
                    decl + r"\section*{Data and code availability}")
(PAPER / "main_elsevier.tex").write_text(pre + body.replace(r"\bibliographystyle{unsrt}", r"\bibliographystyle{elsarticle-num}"))
print("wrote paper/main_elsevier.tex")
