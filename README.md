# Interactive UI for rescoring  3DHISTECH Ltd.'s ScriptQuant(SQ) segmentation results

A tool for editing the results of SQ's segmentation results, making it a useful tool for creating ground truth labels for later ML model trainings and validation
___

#### Notes:
Segmentation scenarios for PDL1 and Ki67 stainings

Uses `PyZMQ` for data transferring from the `ScriptQuant` script and the `tkinter` GUI

The segmentation method can be exchanged on the `ScriptQuant` side, currently it uses `Stardist` and color deconvolution

Resulting segmented objects can be added or disabled.

The results can be exported into csv files, internally data is handled with `pandas`

Files that end with `_sq.py` are meant to be copied into the SQ code area.

Files for the GUI should be pasted into '\\ScriptQuant.Examples\\Usecases\\' path in order to work

`ref` directory contains the previous versions of the solution

