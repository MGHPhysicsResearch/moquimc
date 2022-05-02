#ifndef MQI_BEAM_MODULE_H
#define MQI_BEAM_MODULE_H

/// \file
///
/// Top abstraction to interpret DICOM beam modules, RT and RT-Ion.

#include <moqui/base/mqi_dataset.hpp>

namespace mqi
{

/// \class beam_module
///
/// A class that interprets IonControlPoints and converts fluence_map to be actually delivered.
/// beam_module        : provides overall interface to MC engine
/// beam_module_rt     : photon/electron  -> DYNAMIC, STATIC
/// beam_module_rtion  : proton/particles -> UNIFORM, MODULATED, MODULATED_SPEC
class beam_module
{
public:
protected:
    const mqi::dataset*      ds_;   ///< Item of IonBeamSequence
    const mqi::modality_type modality_;

public:
    beam_module(const mqi::dataset* d, mqi::modality_type m) : ds_(d), modality_(m) {
        ;
    }
    ~beam_module() {
        ;
    }
    virtual void
    dump() const {
        ;
    }
};

}   // namespace mqi

#endif
