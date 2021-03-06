py_files := $(wildcard notebooks/*.py)
ipynb_files := $(wildcard notebooks/*.ipynb)
py_run_targets := $(addprefix run__, $(py_files))
ipynb_validate_targets := $(addprefix validate__, $(ipynb_files))
ipynb_regenerate_targets := $(addprefix regenerate__, $(addsuffix .ipynb, $(basename $(py_files))))
py_regenerate_targets := $(addprefix regenerate__, $(addsuffix .py, $(basename $(ipynb_files))))
PYTHON := python

test: $(ipynb_validate_targets)
run_notebooks: $(py_run_targets)
validate_notebooks: $(ipynb_validate_targets)
regenerate_ipynb: $(ipynb_regenerate_targets)
regenerate_py: $(py_regenerate_targets)

$(py_run_targets): run__%.py :
	[ -e $*.py.skip ] || $(PYTHON) $*.py

$(ipynb_validate_targets): TEMPFILE := $(shell mktemp)
$(ipynb_validate_targets): validate__%.ipynb :
	nbencdec encode $*.ipynb $(TEMPFILE)
	diff -q <(cat $*.py | egrep -v '^# EPY: stripped_notebook: ') <(cat $(TEMPFILE) | egrep -v '# EPY: stripped_notebook: ')

$(ipynb_regenerate_targets): regenerate__%.ipynb : %.py
	nbencdec decode $*.py $*.ipynb

$(py_regenerate_targets): regenerate__%.py : %.ipynb
	nbencdec encode $*.ipynb $*.py
