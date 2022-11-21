#pragma once

// User-changeable parameters  (has original setter function)
#define external_expectedInsertFrac 1
#define external_maxNodeSize 1 << 24
#define external_approximateModelComputation true
#define external_approximateCostComputation false

//Experimental parameters (may break the system)
#define external_fanoutSelectionMethod 0
#define external_splittingPolicyMethod 1
#define external_allowSplittingUpwards false

//Constant parameters in ALEX
#define external_kMinOutOfDomainKeys 5
#define external_kMaxOutOfDomainKeys 1000
#define external_kOutOfDomainToleranceFactor 2

#define external_kMaxDensity 0.8
#define external_kInitDensity 0.7
#define external_kMinDensity 0.6
#define external_kAppendMostlyThreshold 0.9