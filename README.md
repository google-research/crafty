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


**Crafty-RxR data**

Crafty-RxR is built from the [RxR dataset](https://github.com/google-research-datasets/RxR), wherein the instructions are replaced with Crafty-generated ones using the RxR paths. The data are saved as JSONs with dict, with the following data fields:

* split: The annotation split (train/val_seen/val_unseen);
* path_id: Uniquely identifies a path sampled from the Matterport3D environment;
* scan: Uniquely identifies a scan in the Matterport3D environment;
* path: A sequence of panoramic viewpoints along the path;
* distance: The distance traveled in the navigation path.
* heading: The initial heading in radians. Following R2R, the heading angle is zero facing the y-axis with z-up, and increases by turning right.
* instructions: The navigation instructions (3 instructions per path).

A sample entry is as follows

```
{"split": "crafty_RxR_val_seen",
 "distance": 4.2938162285995976,
 "scan": "rPc6DW4iMge",
 "path_id": 28945,
 "path": ["acbe920a0c5d4c018b1803ee9b1f331a",
          "cabddb2506ee4bfcb2a1bcfa6625fe28",
          "30030a9a530b40fc88825ca8fc32f855",
          "f378f8971f7b41ccb0f2c1dc14ba290d"],
 "heading": 0.6938930811500333,
 "instructions": [
     "you should see a door frame slightly right of you. bear right, so that it is in front of you. travel straight and go right, passing the door to the left. head forward, going along to the picture in front of you. face left. travel straight. stop when you get near the stair.",
     "a door frame is a little on the right. curve right, so that it is ahead of you. go straight and turn right, passing the door to your left. you'll see a picture in front of you as you head straight. go left. a picture is on the right. walk straight to the stair and stop there.",
     "a door frame is just on the right. go slightly right, so that it is ahead of you. proceed forward and make a right, passing the door on your left. walk forward, going along to the picture in front of you. you'll see a picture on your right as you go left. proceed forward, then wait by the stair."]}
```

Some statistics on the data:

crafty_RxR_train      | File size: 46M (en); 120M (hi); 137M (te) | #Entries: 19,056
crafty_RxR_val_seen   | File size: 5M (en); 15M (hi); 17M (te)    | #Entries: 2,365
crafty_RxR_val_unseen | File size: 7M (en); 20M (hi); 23M (te)    | #Entries: 3,372

The dataset is downloaded via
```
gsutil -m cp -R gs://crafty-rxr-data .
```

# License
The Matterport3D dataset is governed by the
[Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf).
Crafty is released under the Apache 2.0 license.

# Disclaimer
Not an official Google product.
