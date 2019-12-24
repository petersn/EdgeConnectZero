// SWIG header for the runtime system.

%include <stdint.i>
%include <std_string.i>
%include <std_pair.i>

%module edgeconnect_rules
%include <std_vector.i>
%{
    #include "edgeconnect_rules.h"
%}

%include "edgeconnect_rules.h"

namespace std {
    %template(vectori) vector<int>;
    %template(pairii) pair<int, int>;
}

