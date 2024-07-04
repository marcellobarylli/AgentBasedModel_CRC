from graphviz import Source
dot = Source.from_file('Digraph.gv')

from graphviz import Digraph


dot.render('Digraph')