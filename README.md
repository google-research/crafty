# Crafty

Crafty is a templatic navigation instruction generator which is built using
the data and annotations from the [Matterport3D dataset](https://niessner.github.io/Matterport/)
and the [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator).


**Components of Crafty**

Crafty has four main components.

* Appraise. Given an object, Crafty assesses how interesting it is. Currently, this is done based on how unique its category is (e.g. a couch is more interesting than a wall or ceiling), but it could also be based on its size, color and other attributes.
* Walk. Given a panorama sequence with a starting heading in the first panorama, Crafty computes the sequence of moves necessary to get to the end. This includes the panoramas visited and the entry and exit headings as one moves from one panorama to the next.
* Observe. Given these moves, Crafty then chooses a sequence of objects which are salient during each step of the journey. This is based on computing uniqueness of objects across all environments and their visibility (based on heading and distance). The object sequence is inferred from a Hidden Markov Model that is constructed using a series of heuristics for both emission and transition distributions.
* Talk. Finally, these moves and observations are provided to a templatic language generator. The generator handles the first and final observations as special cases, and then it attempts to provide a mix of movement and object descriptions for the path in between. Multiple templates are defined for each situation, and one of these is randomly chosen for a given portion of a particular path.

Example usage: running Crafty to create instructions for paths from the `val_seen` split of the Room-Across-Room (R2R) dataset.

```
python3 ~/crafty/src/create_instructions.py \
  --path_input_dir=/path/to/matterport_dir \
  --dataset=R2R \
  --file_identifier=val_seen \
  --output_file ~/crafty_R2R_val_seen.json
```

The output will be `crafty_R2R_val_seen.json` file which contains a list of
data dicts each of which is of the same format as the input (`R2R_val_seen`
in this case), except the `instructions` field contains Crafty generated
templatic instructions.


**Dependencies**

* Python-general dependencies (can be easily installed with `pip` or `conda`):

  These are listed under `requirements.txt`, run `pip3 install -r requirements.txt`.

* Specialized package dependencies (requires installation through Github repos):
  * [Apache Beam](https://github.com/apache/beam)
  * [VALAN](https://github.com/google-research/valan)

  The installation configs are described in detail in their `README.md`.



**Data Setup and Source**

To set up the dataset
to work with Crafty, please structure the data directory as follows:

```
base_dir/
  data/
    v1/
      scans/
      connections/
```

* connections: Navigation graphs from the `connectivity` directory in [Matterport3DSimulator github](https://github.com/peteanderson80/Matterport3DSimulator).
* scans: Obtained through the [Matterport3D github](https://github.com/niessner/Matterport) by contacting the owners at matterport3d@googlegroups.com (cf. `Data` section in their `README.md`).

Note: If you would like to structure your data differently, please refer
to the `__init__()` in the class `MatterportData` in `mp3d.py`. For example, you
may remove the directory level `v1` (this was created in the expectation that
there might be more versions).


**Configurable Parameters**

Crafty has several parameters that modify which objects it tends to focus on, including how much it focuses primarily on objects directly ahead of it, how much it prefers to consider objects near or far, how much it cares about object uniqueness and how much it tends to linger on a single object over multiple steps.

The parameters that can be played with can be found in `observe.py` (top of
the script), e.g. `_GAMMA_SHAPE_DISTANCE` adjusts how much do you want
Crafty to favor close-by objects (more details can be found in `observe.py`
docstrings).


# License
The Matterport3D dataset is governed by the
[Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).
Crafty is released under the Apache 2.0 license.

# Disclaimer
Not an official Google product.
