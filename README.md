# maniSourceLocalize

This repository contains information and some implemented procedures regarding the problem of acoustic source localization using a promising, learning-based technique that adapts to the acoustic environment. In particular, we look at the scenario when a node in the acoustic sensor network (ASN) is displaced from the position it was in during training. Consequently, the estimation outcome of the trained localization model in these situations is also impaired, therefore implying the need for a method to detect these compromised nodes. We propose a method that considers the disparity in position estimates made by leave-one-node-out (LONO) sub-networksand uses a Markov random field framework to infer the probability of each LONO position estimate being aligned, misaligned or unreliable with respect to the noise inherent to the estimator. This probabilistic approach is advantageous over naıve detection methods, as it outputs a normalized value that encapsulates conditional information provided by each LONO indicating if the reading is in misalignment with the overall network. Experimental results confirm that the performance of the proposed method is consistent inidentifying compromised nodes for different levels of noise and re-verberation in a given acoustic environment. We compare these results with two naive detection methods, a method that compares the localization efforts of single nodes, and by LONO sub-networks of nodes.

One can either run any script ending in "_vis.m" to see the results of pre-simulated room environments. Otherwise, one can begin with the file trainRoom_rtf_sim.m to simulate their own environment. Please take care that the file names where new simulations are stored are updated. To run the file a mex compiler is needed.

Some of the different visualizations that can be seen include:

sourceLocalize_vis.m: visualize an SSGP localization effort (based off specifications of a room and rtfs learned in trainRoom_rtf_sim.m).

trainRoom_VariVis.m: visualize the mean, variation in error of the SSGP process in specified conditions.

residualDistribution_vis.m: visualize the residual error for both the aligned (no nodes move) and misaligned case (one random node moves).

moveDetect_motive_vis.m: visualize the mean and std. of the localization error for the full ASN, estimation efforts by single nodes and LONO networks when a random array is moved incrementally away from where it was when room dynamics were learned.

moveDetect_paramOpt_vis.m: visualize via heatmap, performance of MRF-detector for various hyperparameter choices related to the aligned, misaligned distributions (normal, exponential respecttively).

moveDetect_gt_vis.m: visualize the ROC curve with respect to the different detection methods.

Information regarding the procedure is commented in the file, and more information can be found in the paper, MISALIGNMENT RECOGNITION IN ACOUSTIC SENSOR NETWORKS USING ASEMI-SUPERVISED SOURCE ESTIMATION METHOD AND MARKOV RANDOM FIELDS.