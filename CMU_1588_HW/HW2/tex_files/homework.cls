\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{homework}[Homework style file]
\LoadClass{article}

\makeatletter
\RequirePackage[utf8]{inputenc} % allow utf-8 input
\RequirePackage[T1]{fontenc}    % use 8-bit T1 fonts
\RequirePackage{hyperref}       % hyperlinks
\hypersetup{
    unicode=true,           % non-Latin characters in Acrobat's bookmarks
    pdffitwindow=false,     % window fit to page when opened
    pdfstartview={FitH},    % fits the width of the page to the window
    %pdftitle={My title},   % title
    %pdfkeywords={keyword1} {key2} {key3}, % list of keywords
    colorlinks=true,        % false: boxed links; true: colored links
    linkcolor=red,          % color of internal links
                            % (change box color with linkbordercolor)
    citecolor=magenta,      % color of links to bibliography
    filecolor=blue,         % color of file links
    urlcolor=blue           % color of external links
}
\RequirePackage{url}            % simple URL typesetting
\RequirePackage{booktabs}       % professional-quality tables
\RequirePackage{amsfonts}       % blackboard math symbols
\RequirePackage{nicefrac}       % compact symbols for 1/2, etc.
\RequirePackage{microtype}      % microtypography
\RequirePackage{float}
\RequirePackage{multicol}
\RequirePackage{multirow}
\RequirePackage{siunitx}
\RequirePackage{tikz}
\usetikzlibrary{calc,shapes,arrows,positioning,arrows.meta,matrix}
\RequirePackage{pgf}
\RequirePackage{mleftright}
\RequirePackage{mdframed}
\RequirePackage{enumitem}
\RequirePackage{lmodern}
\RequirePackage{fullpage}
\RequirePackage{natbib}
\RequirePackage{lineno}
\RequirePackage{etoolbox}
\RequirePackage{amsmath}
% \RequirePackage{times}
% \linenumbers
% fix incompatibilities between lineno and amsmath, if required, by
% transparently wrapping linenomath environments around amsmath
% environments
\AtBeginDocument{%
    \newcommand*\patchAmsMathEnvironmentForLineno[1]{%
      \expandafter\let\csname old#1\expandafter\endcsname\csname #1\endcsname
      \expandafter\let\csname oldend#1\expandafter\endcsname\csname end#1\endcsname
      \renewenvironment{#1}%
                       {\linenomath\csname old#1\endcsname}%
                       {\csname oldend#1\endcsname\endlinenomath}%
    }%
    \newcommand*\patchBothAmsMathEnvironmentsForLineno[1]{%
      \patchAmsMathEnvironmentForLineno{#1}%
      \patchAmsMathEnvironmentForLineno{#1*}%
    }%
    \patchBothAmsMathEnvironmentsForLineno{equation}%
    \patchBothAmsMathEnvironmentsForLineno{align}%
    \patchBothAmsMathEnvironmentsForLineno{flalign}%
    \patchBothAmsMathEnvironmentsForLineno{alignat}%
    \patchBothAmsMathEnvironmentsForLineno{gather}%
    \patchBothAmsMathEnvironmentsForLineno{multline}%
}
\RequirePackage{bm}
\RequirePackage{amsthm}
% \newtheorem{proposition}[]{Proposition}
% \newtheorem{lemma}[]{Lemma}
% \newtheorem{definition}[]{Definition}
% \newtheorem{property}[]{Property}
% \newtheorem{example}[]{Example}
% \newtheorem{theorem}[]{Theorem}
% \newtheorem{corollary}[]{Corollary}
\RequirePackage{thm-restate}
\RequirePackage{amssymb}
\RequirePackage{mathtools}
\RequirePackage{letltxmacro}
\RequirePackage{environ}
\newcommand{\Rp}{\bbR_{\ge 0}}

\newcommand{\sfbf}[1]{\textbf{\sffamily#1}}
\RequirePackage{titlesec}

\titleformat{\part}[display]
    {\normalfont\filcenter}
    {\LARGE\sfbf{Part~\thepart}}
    {1pc}
    {\vspace{1pc}\Huge\bfseries\sffamily}
    [\newpage]

\titleformat{\chapter}[display]
    {\normalfont\filright} % format
    {\LARGE\sfbf{\chaptertitlename~\thechapter}} % label
    {1pc} % sep between label and title body
    {\Huge\bfseries\sffamily} % before-code
\titlespacing*{\chapter}{0pt}{40pt}{50pt}

\titleformat{\section}
{\normalfont\Large\bfseries\sffamily}{\thesection}{1em}{}
\titleformat{\subsection}
{\normalfont\large\bfseries\sffamily}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
{\normalfont\normalsize\bfseries\sffamily}{\thesubsubsection}{1em}{}
\titleformat{\paragraph}[runin]
{\normalfont\normalsize\bfseries\sffamily}{\theparagraph}{1em}{}

\newcommand{\hint}[1]{\footnotetext{$\bigstar$ Hint: #1}}

% % Rename proof => baseproof
% \GlobalLetLtxMacro{\baseproof}{\proof}
% \GlobalLetLtxMacro{\endbaseproof}{\endproof}

% % Introduce new shadedproof environment
% \renewenvironment{proof}[1][\relax]{%
%     \begin{mdframed}[linecolor=gray,innerleftmargin=4mm,innerrightmargin=0mm,topline=false,bottomline=false,rightline=false,backgroundcolor=white]
%         \ifx#1\relax\begin{baseproof}\else\begin{baseproof}[#1]\fi
% }{\end{baseproof}\end{mdframed}}

\newenvironment{solution}{%
    \begin{mdframed}[linecolor=gray,innerleftmargin=4mm,innerrightmargin=0mm,topline=false,bottomline=false,rightline=false,backgroundcolor=white]
        \begin{proof}[Solution]
}{\end{proof}\end{mdframed}}


\RequirePackage[capitalise,noabbrev]{cleveref}

\newtheoremstyle{thmstyle}
  {1mm} % Space above
  {2mm} % Space below
  {} % Body font
  {} % Indent amount
  {\sffamily\bfseries} % Theorem head font
  {.} % Punctuation after theorem head
  {.5em} % Space after theorem head
  {} % Theorem head spec (can be left empty, meaning `normal')

\theoremstyle{thmstyle}
% \newtheorem{assumption}{Assumption}[]
\newtheorem{mdxcorollary}{Corollary}
\newtheorem{mdxdefinition}{Definition}
\newtheorem{mdxexample}{Example}
\newtheorem{mdxfact}{Fact}
\newtheorem{mdxquestion}{Question}
\newtheorem{mdxlemma}{Lemma}
\newtheorem{mdxproposition}{Proposition}
\newtheorem{mdxremark}{Remark}
\newtheorem{mdxtheorem}{Theorem}
\newtheorem{mdxobservation}{Observation}
\newtheorem{mdxopenproblem}{Open problem}
\newtheorem{mdxproblem}{Problem}[section]


\newenvironment{definition}[1][\relax]%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\ifx#1\relax\begin{mdxdefinition}\else\begin{mdxdefinition}[#1]\fi}%
  {\end{mdxdefinition}\end{mdframed}\medskip}
\newenvironment{example}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxexample}}%
  {\end{mdxexample}\end{mdframed}\medskip}
\newenvironment{question}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxquestion}}%
  {\end{mdxquestion}\end{mdframed}\medskip}
\newenvironment{fact}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxfact}}%
  {\end{mdxfact}\end{mdframed}\medskip}
\newenvironment{theorem}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxtheorem}}%
  {\end{mdxtheorem}\end{mdframed}\medskip}
\newenvironment{lemma}[1][\relax]%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\ifx#1\relax\begin{mdxlemma}\else\begin{mdxlemma}[#1]\fi}%
  {\end{mdxlemma}\end{mdframed}\medskip}
\newenvironment{proposition}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxproposition}}%
  {\end{mdxproposition}\end{mdframed}\medskip}
\newenvironment{remark}[1][\relax]%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\ifx#1\relax\begin{mdxremark}\else\begin{mdxremark}[#1]\fi}%
  {\end{mdxremark}\end{mdframed}\medskip}
\newenvironment{corollary}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxcorollary}}%
  {\end{mdxcorollary}\end{mdframed}\medskip}
\newenvironment{observation}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxobservation}}%
  {\end{mdxobservation}\end{mdframed}\medskip}
\newenvironment{openproblem}%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\begin{mdxopenproblem}}%
  {\end{mdxopenproblem}\end{mdframed}\medskip}
\newenvironment{problem}[1][\relax]%
  {\medskip\begin{mdframed}[linecolor=gray!100,backgroundcolor=gray!0]\ifx#1\relax\begin{mdxproblem}\else\begin{mdxproblem}[#1]\fi}%
  {\end{mdxproblem}\end{mdframed}\medskip}
%%% Algorithm configuration
\RequirePackage[linesnumbered,ruled,lined,noend]{algorithm2e}
% Fix for line number alignment: https://tex.stackexchange.com/questions/476579/how-to-align-line-numbers-horizontally-with-package-algorithm2e
\patchcmd\algocf@Vline{\vrule}{\vrule \kern-0.4pt}{}{}
\patchcmd\algocf@Vsline{\vrule}{\vrule \kern-0.4pt}{}{}

\SetKwComment{Hline}{}{\vspace{-3mm}\textcolor{gray}{\hrule}\vspace{1mm}}
\definecolor{darkgrey}{gray}{0.3}
\definecolor{commentcolor}{gray}{0.5}
\SetKwComment{Comment}{\color{commentcolor}[$\triangleright$\ }{}
\SetCommentSty{}
\SetNlSty{}{\color{darkgrey}}{}
\setlength{\algomargin}{4mm}
\SetKwProg{Fn}{function}{}{}
\SetKwProg{Subr}{subroutine}{}{}
\crefalias{AlgoLine}{line}%
\crefname{algocf}{Algorithm}{Algorithms}
\crefname{mdxproblem}{Problem}{Problems}



\let\cref@old@stepcounter\stepcounter
\def\stepcounter#1{%
  \cref@old@stepcounter{#1}%
  \cref@constructprefix{#1}{\cref@result}%
  \@ifundefined{cref@#1@alias}%
    {\def\@tempa{#1}}%
    {\def\@tempa{\csname cref@#1@alias\endcsname}}%
  \protected@edef\cref@currentlabel{%
    [\@tempa][\arabic{#1}][\cref@result]%
    \csname p@#1\endcsname\csname the#1\endcsname}}

\crefname{property}{Property}{Properties}
\crefname{example}{Example}{Examples}

\RequirePackage{xcolor}

\newcommand*\circled[1]{\tikz[baseline=(char.base)]{
            \node[shape=circle,draw,inner sep=1pt] (char) {\normalfont\small #1};}}
\newcommand{\numberthis}[1]{\refstepcounter{equation}\tag{\theequation}\label{#1}}

\newcommand{\course}[1]{\gdef\@course{#1}}
\newcommand{\homework}[1]{\gdef\@homework{#1}}
\newcommand{\student}[1]{\gdef\@student{#1}}
\newcommand{\email}[1]{\gdef\@email{#1}}
\newcommand{\releasedate}[1]{\gdef\@releasedate{#1}}
\newcommand{\duedate}[1]{\gdef\@duedate{#1}}
\newcommand{\instructor}[1]{\gdef\@instructor{#1}}
\newcommand{\envelope}{\raisebox{-.5mm}{\tikz[xscale=.37,yscale=.5]{\draw (0,.5) -- (0,0) -- (1,0) -- (1,.5) -- (.5,.23) -- (0,.5) -- (1, .5);}}}
\renewcommand{\maketitle}{%
    \begin{mdframed}
        \@course \hfill Released on: \@releasedate\\~\\~\\
        \centering{\sffamily\Large Homework \@homework}
        \\~\\~\\
      \raggedright Student: \@student \hfill \textbf{Due on: \@duedate}
    \end{mdframed}
    \vspace{1cm}
}
\makeatother

\newcommand{\runinsec}[1]{\medskip\noindent\textbf{#1.}~}