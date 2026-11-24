COLORS = ('red', 'blue', 'purple', 'yellow', 'white')
SHAPES = ('cube', 'bar', 'cylinder', 'triangle', 'arch', 'stick')

PLAN = """Task: Put every object on the table into a region, with one object per region.
Any one-to-one assignment of objects to regions is a valid reset; a region's color does not need to match the object's color.
Objects to reset: {objects}
Use the current top-camera RGB image (first image), the wrist-camera image (second image) and your own previous plans and execution results to decide the next object and its destination region.
Only one object may be operated on in this step.
If the task is complete, set all_done to true.

Destination region names: {regions}.

Return only one JSON object with exactly these fields:
{{
  "next": "<one name from Objects to reset>",
  "region": "<destination region name>",
  "all_done": false,
  "reason": "<your reason>"
}}
When all_done is false, next must be exactly one of the names in Objects to reset, and region must be one destination region name.
When all_done is true, next and region must both be null.

Your own previous plans and execution results:
{history}"""

LOCATE = """The image is {w}x{h} pixels; tick marks and numbers along the edges show pixel coordinates every 64 pixels.
Find each of these in the image and give the pixel coordinates of its center: {items}.
If one of them is not visible, give null for it.
Return only one JSON object mapping each name to {{"x": <number>, "y": <number>}} or null."""

REFINE = """This is a {w}x{h} zoomed crop of a camera image; tick marks and numbers along the edges show crop pixel coordinates every {step} pixels.
Find the {name} in this crop and give the pixel coordinates of its center in the crop.
If it is not in the crop, return {{"x": null, "y": null}}.
Return only one JSON object: {{"x": <number or null>, "y": <number or null>}}"""

ATTR = 'Choose exactly one color and one shape for the object at the center of the image.\nCOLORS: ["red", "blue", "purple", "yellow", "white"]\nSHAPES: ["cube", "bar", "cylinder", "triangle", "arch", "stick"]\nReturn only one JSON object with exactly these fields: {"color": "<one word from COLORS>", "shape": "<one word from SHAPES>"}.'

RESELECT = """The robot must pick up the {obj}. The point located for the {obj} lies on an object that is already in its region.
Each of the following {k} images is a zoomed crop of the top camera centered on another object on the table (image 1 to image {k}).
Which image shows the {obj} at its center? If none does, answer null.
Return only one JSON object: {{"index": <integer from 1 to {k}, or null>}}"""

MEMORYCHECK = """The first image is a zoomed crop of the top camera taken right after the {obj} was placed in the {region}.
The second image is a zoomed crop of the same location now.
Is the {obj} still in the same place, at the center of the second image?
Return only one JSON object: {{"still_there": true or false}}"""
