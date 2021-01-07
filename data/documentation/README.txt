
(C) Algorithmic Engineering group, Universita' di Roma "La Sapienza"
    Funded by the DELIS project - Released in August 2006.

This collection is the result of the effort of a team of volunteers:

       Thiago Alves                 Alex Ntoulas
       Luca Becchetti               Josiane-Xavier Parreira
       Paolo Boldi                  Xiaoguang Qi
       Paul Chirita                 Massimo Santini
       Mirel Cosulschi              Tamas Sarlos
       Brian Davison                Mike Thelwall
       Pascal Filoche               Belle Tseng
       Antonio Gulli                Tanguy Urvoy
       Zoltan Gyongyi               Wenzhong Zhao
       Thomas Lavergne

The collection was downloaded in May 2006 by the Laboratory of Web
Algorithmics, Universita' degli Studi di Milano. The labelling
process was coordinated by Carlos Castillo.

========================================================================
webspam-uk2006-labels.txt - The labels themselves.

The labels are contained in a plain text file (webspam-uk2006-labels.txt)
with one line per host. On each line, there are four fields separated
by spaces:

 - Field #1: host name.
 - Field #2: judgments.
 - Field #3: "spamicity" measure.
 - Field #4: label for the host.

The judgments are a comma-separated list of "judge:judgment" pairs.
Human judges are identified by 'jNN', in which NN are identifiers
assigned to the judges in an arbitrary order. The credits at the
beginning of this file mention only the names of the judges that
classified 200 hosts or more.

We also added two special judges: 'odp' and 'domain'.  The judge
'odp' labels as normal all the UK hosts that were mentioned in
the Open Directory Project (http://www.dmoz.org/) on May 2006.
The judge 'domain' labels as normal all the UK hosts ending in
.ac.uk, .sch.uk, .gov.uk, .mod.uk, .nhs.uk or .police.uk.

The spamicity measure is calculated by assigning 1 point for each
'spam' (S) judgment, 0.5 points for each 'borderline' (B)
judgment, and 0 points for each 'normal' (N) judgment, and taking
the average.  The 'can not judge' (?) marks are ignored for this
calculation.

The label for a host with an average of over 0.5 is spam, for a
host with an average of less than 0.5 is normal, and for a host
with exactly 0.5 is undecided.

========================================================================

More information is available at http://aeserver.dis.uniroma1.it/webspam/
including the guidelines given to the human judges, the instructions
for obtaining the links and contents of the pages in this collection,
and the contact information for questions and comments.

--
Carlos Castillo
August 2006
