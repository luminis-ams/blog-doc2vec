# A minimal PV-DBOW example implementation

Code acompanying this blog post:
https://amsterdam.luminis.eu/2017/02/21/coding-doc2vec/

And two previous blog posts leading up to the abovementioned one:
- https://amsterdam.luminis.eu/2017/01/16/googling-doc2vec/
- https://amsterdam.luminis.eu/2017/01/30/implementing-doc2vec/

As a starting point for the code, the word2vec implementation in
assignment 5 was used, and lots of small things were subsequently
changed, added, and removed.

Feel free to play with this code, improve it, fix it, and add to it.

## Installation

Install Python 3, and the packages listed at the top of the
`.ipynb` file. And finally Jupyter Notebook.

Remember, before you run it, to set `PERCENTAGE_DOCS` to a lower value, e.g.,
something like 5 percent, just to check if it all works before you train a
network on the whole Reuters training data set (which may take a while).
