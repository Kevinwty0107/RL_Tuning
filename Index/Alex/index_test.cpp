#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <string>
#include <sstream>

using namespace std;

#include "src/alex_map.h"
// #include "src/alex_multimap.h"

template<typename T>
bool load_binary_sosd(string filename, vector<T> &v)
{
    ifstream ifs(filename, ios::in | ios::binary);
    assert(ifs);

    T size;
    ifs.read(reinterpret_cast<char*>(&size), sizeof(T));
    v.resize(size);
    ifs.read(reinterpret_cast<char*>(v.data()), size * sizeof(T));
    ifs.close();

    return ifs.good();
}

int main(int argc, char * argv[]) 
{
    //SOSD data path
    string data_file_path = argv[1];
    
    //Load Data
    vector<uint64_t> data;
    if (!load_binary_sosd(data_file_path,data))
    {
        cout << "input stream status error" << endl;
    }

    //If we want to sort the data
    sort(data.begin(),data.end());

    //Load ALEX
    alex::Alex<uint64_t,int> alex;

    //Load into ALEX (No Bulk Load)
    for (int i = 0; i < 100; ++i)
    {
        alex.insert(data[i],i);
    }

    // //Load into ALEX (bulk load)
    // //There is a bug when array gets too large
    // const int num_key = data.size();
    // pair<uint64_t,int> dataArray[num_key];
    // for (int i = 0; i < num_key; ++i)
    // {
    //     dataArray[i] = make_pair(data[i],i);
    // }
    // alex.bulk_load(dataArray,num_key);
    
    // Run Query 
    clock_t start,end;
    start=clock();
    for (int query_key = 0; query_key < 1000; ++query_key)
    {
        alex.count(query_key);
        alex.find(query_key);
        alex.lower_bound(query_key);
        alex.upper_bound(query_key);
    }
    end=clock();

    //Output into File
    ofstream run_time_out("./runtime_result.txt");
    assert(run_time_out);
    run_time_out << (double)(end-start)/CLOCKS_PER_SEC << endl;
    run_time_out.close();
    if (!run_time_out.good())
    {
        cout << "runtime_result out stream status error" << endl;
    }

    ofstream  state_out("./state_result.txt");
    assert(state_out);
    state_out << "no_model_nodes:" << alex.stats_.num_model_nodes << endl;
    state_out << "no_model_node_expansions:" << alex.stats_.num_model_node_expansions << endl;
    state_out << "no_model_node_split:" << alex.stats_.num_model_node_splits << endl;
    state_out << "num_model_node_expansion_pointers:" << alex.stats_.num_model_node_expansion_pointers << endl;
    state_out << "num_model_node_split_pointers:" << alex.stats_.num_model_node_split_pointers << endl;
    state_out << "no_data_nodes:" << alex.stats_.num_data_nodes << endl;
    state_out << "no_expand_and_scale:" << alex.stats_.num_expand_and_scales << endl;
    state_out << "no_expand_and_retrain:" << alex.stats_.num_expand_and_retrains << endl;
    state_out << "no_downward_split:" << alex.stats_.num_downward_splits << endl;
    state_out << "no_sideways_split:" << alex.stats_.num_sideways_splits << endl;
    state_out << "no_downward_split_keys:" << alex.stats_.num_downward_split_keys << endl;
    state_out << "no_sideways_split_keys:" << alex.stats_.num_sideways_split_keys << endl;
    state_out << "no_search:" << alex.stats_.num_lookups << endl;
    state_out << "no_inserts:" << alex.stats_.num_inserts << endl;
    state_out << "no_node_traveral:" << alex.stats_.num_node_lookups << endl;
    state_out.close();
    if (!state_out.good())
    {
        cout << "state_result out stream status error" << endl;
    }
    return 0;
}