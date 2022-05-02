#ifndef MQI_BEAM_MODULE_RT_H
#define MQI_BEAM_MODULE_RT_H


/// \file
/// Interprets DICOM-RT (photon and electron) beam module
/// \see http://dicom.nema.org/dicom/2013/output/chtml/part03/sect_C.8.html#sect_C.8.8.14 for RT Plan
/// \note no plans to implement yet (July 2019).

#include <moqui/base/mqi_beam_module.hpp>

namespace mqi {

/// class beam_module_rt
class beam_module_rt : public beam_module{
public: 
    beam_module_rt(
        const mqi::dataset* d,
        mqi::modality_type m)
        :beam_module(d,m)
    {;}
    ~beam_module_rt(){;}

};

}

#endif
