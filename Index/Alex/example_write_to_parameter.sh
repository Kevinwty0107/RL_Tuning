#Writing to the parameter file

#Read from txt file  
#...

#Write to parameters
> src/parameters.hpp;
echo "#pragma once" >> src/parameters.hpp;
echo "#define external_expectedInsertFrac 1" >> src/parameters.hpp;
echo "#define external_maxNodeSize 1 << 24" >> src/parameters.hpp;
echo "#define external_approximateModelComputation true" >> src/parameters.hpp;
echo "#define external_approximateCostComputation false" >> src/parameters.hpp;
echo "#define external_fanoutSelectionMethod 0" >> src/parameters.hpp;
echo "#define external_splittingPolicyMethod 1" >> src/parameters.hpp;
echo "#define external_allowSplittingUpwards false" >> src/parameters.hpp;
echo "#define external_kMinOutOfDomainKeys 5" >> src/parameters.hpp;
echo "#define external_kMaxOutOfDomainKeys 1000" >> src/parameters.hpp;
echo "#define external_kOutOfDomainToleranceFactor 2" >> src/parameters.hpp;
echo "#define external_kMaxDensity 0.8" >> src/parameters.hpp; 
echo "#define external_kInitDensity 0.7" >> src/parameters.hpp; 
echo "#define external_kMinDensity 0.6" >> src/parameters.hpp; 
echo "#define external_kAppendMostlyThreshold 0.9" >> src/parameters.hpp;

#Compile
#...