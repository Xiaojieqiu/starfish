#!/usr/bin/env python
# coding: utf-8
#
# EPY: stripped_notebook: {"metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.6.5"}}, "nbformat": 4, "nbformat_minor": 2}

# EPY: START markdown
# ## Loading the data into Starfish
# EPY: END markdown

# EPY: START code
from starfish.io import Stack
from starfish.image import ImageStack
import os
# EPY: END code

# EPY: START markdown
# Below, you can grab your data in the current spec (v0.0.0). I've also converted and uploaded the other two fovs (002, 003). I wasn't sure waht genes your codebooks referred to, so they have bogus gene names. 
# 
# These data sets are each about 1gb, so they take some time to load.
# EPY: END markdown

# EPY: START code
s = Stack.from_experiment_json('https://dmf0bdeheu4zf.cloudfront.net/20180608/osmFISH/fov_001/experiment.json')
# EPY: END code

# EPY: START markdown
# If you want to save them locally for faster iteration, load them once above, then write them locally with `s.write` and then load them up from the written object
# EPY: END markdown

# EPY: START code
# os.makedirs('osmFISH', exist_ok=True)
# s.write('osmFISH')
# EPY: END code

# EPY: START code
# s = Stack.from_experiment_json(os.path.join('osmFISH/', 'experiment.json'))
# EPY: END code

# EPY: START markdown
# Below, we list a few of the modules (by category) that we think will be useful for analysis of osmFISH data
# EPY: END markdown

# EPY: START markdown
# ## Filtering
# EPY: END markdown

# EPY: START code
from starfish.pipeline.filter import Filter
# EPY: END code

# EPY: START markdown
# We've already implemented a gaussian high-pass filter.
# EPY: END markdown

# EPY: START code
help(Filter.gaussian_high_pass)
# EPY: END code

# EPY: START code
# Not implemented, but needed for osmFISH
# Filter.gaussian_laplace
# EPY: END code

# EPY: START markdown
# ## Spot calling
# EPY: END markdown

# EPY: START code
from starfish.pipeline.features.spots.detector import SpotFinder

# Not implemented, but needed for osmFISH:
# SpotFinder.peak_local_max 
# EPY: END code

# EPY: START markdown
# You can max project the data like this. I suggest using this function _inside_ your spot calling method. 
# EPY: END markdown

# EPY: START code
help(s.ImageStack.max_proj)
# EPY: END code

# EPY: START markdown
# ## Segmentation
# EPY: END markdown

# EPY: START code
from starfish.pipeline.segmentation import Segmentation
# EPY: END code

# EPY: START code
help(Segmentation.watershed)
# EPY: END code
