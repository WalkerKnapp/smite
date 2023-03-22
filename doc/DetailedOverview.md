## Detailed Overview

***smite*** (Single Molecule Imaging Toolbox Extraordinaire) is a
collection of MATLAB tools developed to process, either through
GUIs or via batch, fluorescent single molecule imaging data.  This
data is typically collected into .h5 (Hierarchical Data Format 5)
files created by the sister software MATLAB Instrument Control
(***MIC***), a collection of MATLAB classes for automated data
collection on complex, multi-component custom built microscopes.
This software can be obtained from the ***MIC*** GitHub distribution
(https://github.com/LidkeLab/matlab-instrument-control.git).

***smite*** is organized into a set of namespaces that group similar
tools and concepts.  The namespace  `+smi`  contains the highest
level tools that will be the most common entry point for processing
SMLM and SPT data sets.

The core functionality of ***smite*** (contained in the namespace
`+smi`) are
- [SMLM](CoreFunctionality/SMLM) (Single Molecule Localization
  Microscopy), which processes 2D super-resolution (SR) data in .h5
  files with standard [contents](FileFormats/H5). Data can also be
  stored in .mat files under a variable with a name like "sequence";;
- [Publish](CoreFunctionality/Publish), which batch-processes SR
  data assuming the .h5 files follow a standard naming convention
  (obj.CoverslipDir/Cell*/Label*/Data*.h5);
- [SPT](CoreFunctionality/SPT) (Single Particle Tracking), which
  analyzes tracking data;
- [BaGoL](CoreFunctionality/BaGoL) (Bayesian Grouping of
  Localizations) explores the possible number of emitters and their
  positions that can explain the observed localizations and
  uncertainties in the data.

Note: A [camera calibration file](FileFormats/CalibrationFile) is
used by SMLM (and hence Publish and SPT) as well.

Corresponding examples are presented in:
- [SMLM](../MATLAB/examples/Example_SMLM_Basic.m)
- [Publish](../MATLAB/examples/Example_Publish.m)
- [SPT](../MATLAB/examples/Example_SPT.m)
- BaGoL:
```
  B=BaGoL()       % create object
  B.SMD=....      % set properties
  B.analyze_all() % run complete analysis
```

In all the examples and throughout ***smite***, three main data
structures are used to organize the analysis parameters and data:
- [SMF](CoreFunctionality/SMF) (Single Molecule Fitting), which
  uniquely and completely defines the data analysis parameters;
- [SMD](CoreFunctionality/SMD) (Single Molecule Data), which stores
  the data results;
- [TR](CoreFunctionality/TR) (Tracking Results), which is a variation
  of SMD used for tracking data results.