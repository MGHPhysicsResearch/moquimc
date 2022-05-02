#ifndef MQI_DATASET_H
#define MQI_DATASET_H

/// \file
///
/// Header of the mqi::dataset class

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "gdcmByteValue.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmItem.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmVM.h"
#include "gdcmVR.h"

namespace mqi
{

/// Enumerate for modality type (0x0008,0x0060)
typedef enum
{
    RTPLAN,      //< RT Plan
    IONPLAN,     //< RT-Ion Plan
    RTRECORD,    //< RT Record
    IONRECORD,   //< RT-Ion Record
    RTIMAGE,     //< RT Image (CT)
    RTSTRUCT,    //< RT Struct
    RTDOSE,      //< RT Dose
    UNKNOWN_MOD
} modality_type;

/// Type of beam id, whether number or name
struct beam_id_type {
    enum
    {
        NUM,
        STR
    } type;
    union {
        int         number;
        const char* name;
    };
};

/// A map for sequential tags per modality types, RTPLAN, IONPLAN, IONRECORD.
static const std::map<const modality_type, const std::map<const std::string, const gdcm::Tag>>
  seqtags_per_modality = {
      //modality, seq_name, tag
      { RTPLAN,
        {
          { "beam", gdcm::Tag(0x300a, 0x00b0) }
          //{"wedge", gdcm::Tag()},
          //{"mlc"  , gdcm::Tag()},
        } },
      {
        IONPLAN,
        { { "beam", gdcm::Tag(0x300a, 0x03a2) },
          { "snout", gdcm::Tag(0x300a, 0x030c) },
          { "rs", gdcm::Tag(0x300a, 0x0314) },     ///< range shifter sequence
          { "rsss", gdcm::Tag(0x300a, 0x0360) },   ///< range shifter setting sequence
          { "blk", gdcm::Tag(0x300a, 0x03a6) },
          { "comp", gdcm::Tag(0x300a, 0x0e2a) },
          { "ctrl", gdcm::Tag(0x300a, 0x03a8) } }   ///< ion control point sequence
      },
      { IONRECORD,
        {
          { "beam", gdcm::Tag(0x3008, 0x0021) },
          { "snout", gdcm::Tag(0x3008, 0x00f0) },
          { "rs", gdcm::Tag(0x3008, 0x00f2) },
          { "rsss", gdcm::Tag(0x300a, 0x0360) },   ///< range shifter setting sequence
          { "blk", gdcm::Tag(0x3008, 0x00d0) },
          { "comp", gdcm::Tag(0x3008, 0x00c0) },
          { "ctrl", gdcm::Tag(0x3008, 0x0041) },
          { "machine", gdcm::Tag(0x300a, 0x0206) }   ///< treatment machine record
        } }
  };

/// \class dataset
/// mqi::data was written to provide the access to gdcm's Nested Dataset,
/// i.e., GetItem() is removed.
/// In RTI, we get values from either tag or keyword.
/// We converts decimal strings to float by default.
/// Example:
/// std::vector<int> tmp;
/// block_ds->get_values("NumberOfBlocks", tmp);
/// or equivalent to
/// block_ds->get_values(gdcm::Tag(0x300a,0x00f0), tmp);
/// Access to an internal dataset. Usual GDCM way.
///
/// GDCM ways were
/// const gdcm::DataElement &beamsq = plan_gdcm_ds_.GetDataElement( gdcm::Tag() );
/// gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = beamsq.GetValueAsSQ();
/// const gdcm::Item &beam = sqi->GetItem(0);
/// In RT-Ion interface, we get dataset directly.
/// Use '(' and ')' for sequence access. return mqi::dataset*
/// Use '[' and ']' for DataElement access
/// Example:
/// const dataset* beam_ds = plan_ds_("IonBeamSequence")
/// or
/// const dataset* beam_ds = plan_ds_(gdcm::Tag(0x300a,0x03a2))
/// \note
/// Known-issues:
///  linker caused error of dupulication because this initialization is done in header...
///   const gdcm::Dicts& dcm_object::dicts_ = gdcm::Global::GetInstance().GetDicts();

class dataset
{
protected:
    /// Tag, Name, and dataset container (recursive)
    std::vector<std::tuple<const gdcm::Tag, const std::string, std::vector<const dataset*>>>
      ds_lut_;

    //const gdcm::DataSet& didn't work. segmentfault
    const gdcm::DataSet gdcm_ds_;   ///< A set of DataElements (attributes)

public:
    /// Fill ds_lut_ for given gdcm::Dataset
    /// \param d gdcm::DataSet
    /// \param include_sq option for reading SQ (sequence) recursively.
    dataset(const gdcm::DataSet& d, const bool include_sq = true) : gdcm_ds_(d) {
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();

        if (!include_sq) { return; }

        for (auto el = gdcm_ds_.Begin(); el != gdcm_ds_.End(); ++el) {
            /// el->getValue() doesn't guarantee it has value.
            const gdcm::Tag&       tag   = el->GetTag();
            const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);

            if (!(entry.GetVR() & gdcm::VR::SQ)) continue;

            gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = el->GetValueAsSQ();
            std::vector<const dataset*>               tmp(0);
            for (size_t i = 1; i <= sqi->GetNumberOfItems(); ++i) {
                const gdcm::Item& itm = sqi->GetItem(i);
                tmp.push_back(new mqi::dataset(itm.GetNestedDataSet()));
            }
            ds_lut_.push_back(std::make_tuple(tag, entry.GetKeyword(), tmp));

        }   //gdcm_ds_
    }

    /// Access to a DataElement with a tag
    /// \param t gdcm::Tag&
    /// \return a constant reference of gdcm::DataElement
    /// \return empty DataElement if not found
    /// \note it seems better to return copy object or reference?
    const gdcm::DataElement&
    operator[](const gdcm::Tag& t) const {
        if (gdcm_ds_.FindDataElement(t) && !gdcm_ds_.GetDataElement(t).IsEmpty()) {
            return gdcm_ds_.GetDataElement(t);
        } else {
            static gdcm::DataElement empty;
            empty.Clear();   //invalid VR & VL
            return empty;
        }
    }

    /// Access to a DataElement with a tag's keyword
    /// \param keyword tag's keyword in string
    /// \return a constant reference of gdcm::DataElement
    /// \return empty DataElement if not found
    const gdcm::DataElement&
    operator[](const char* keyword) const {
        std::string        tmp(keyword);
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();

        for (auto it = gdcm_ds_.Begin(); it != gdcm_ds_.End(); ++it) {
            const gdcm::DictEntry& entry = dicts.GetDictEntry(it->GetTag());
            if (!tmp.compare(entry.GetKeyword())) return (*it);
        }
        static gdcm::DataElement empty;
        empty.Clear();   //invalid VR & VL
        return empty;
    }

    /// Access dataset container with a Tag
    /// \param t gdcm tag
    /// \return a dataset container
    /// \return empty if not found
    /// \note none of these worked.
    /// const std::vector<const dataset*>& operator()(const std::string& s)
    /// std::vector<const dataset*> operator()(const std::string& s)
    /// std::vector<const dataset*> operator()(const std::string& s)
    /// std::vector<const dataset*> operator()(const char* s) const {
    std::vector<const dataset*>
    operator()(const gdcm::Tag& t) const {
        for (auto& i : ds_lut_)
            if (std::get<0>(i) == t) { return std::get<2>(i); }
        std::vector<const dataset*> empty(0);
        return empty;
    }

    /// Access dataset container with a Tag keyword
    /// \param s tag string
    /// \return a dataset container
    /// \return empty if not found
    std::vector<const dataset*>
    operator()(const char* s) const {
        std::string tmp(s);
        for (auto& i : ds_lut_)
            if (!tmp.compare(std::get<1>(i))) return std::get<2>(i);

        std::vector<const dataset*> empty(0);
        return empty;
    }

    /// Destructor
    ~dataset() {
        for (auto& i : ds_lut_) {
            for (auto& j : std::get<2>(i)) {
                delete j;
            }
            std::get<2>(i).clear();
        }
        ds_lut_.clear();
    }

    /// Print out all DataElement in this dataset resursively.
    void
    dump() const {
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();

        for (auto el = gdcm_ds_.Begin(); el != gdcm_ds_.End(); ++el) {
            const gdcm::DictEntry& entry = dicts.GetDictEntry(el->GetTag());
            std::cout << entry.GetKeyword() << ": " << el->GetTag() << ", VR: " << el->GetVR()
                      << ", length: " << el->GetVL() << std::endl;
        }

        std::cout << " LUT: --- size: " << ds_lut_.size() << std::endl;
        for (auto& i : ds_lut_) {
            std::cout << "     (" << std::get<0>(i) << ", " << std::get<1>(i) << ", "
                      << std::get<2>(i).size() << " )";
        }
        std::cout << std::endl;
    }

    /// Get DICOM values in a float vector
    /// \param bv pointer to ByteValue
    /// \param vr DICOM Value Representation: http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html
    /// \param vm DICOM Value Multiplicity: http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.4.html
    /// \param vl Value length
    /// \param res reference to a "float" container for dicom values.
    /// \return void
    void
    get_values(const gdcm::ByteValue* bv,
               const gdcm::VR         vr,
               const gdcm::VM         vm,
               const gdcm::VL         vl,
               std::vector<float>&    res) const {
        res.clear();

        if (vr & gdcm::VR::VRBINARY) {
            assert(vr & gdcm::VR::FL);

            size_t ndim = vl / sizeof(float);   ///< T should be given 'float'
            res.resize(ndim);
            bv->GetBuffer((char*) &res[0], ndim * sizeof(float));
        } else if (vr & gdcm::VR::VRASCII) {
            //ascii & numeric (int, float)
            //ascii & IS, DS, -> decimal strings...
            std::string       s    = std::string(bv->GetPointer(), bv->GetLength());
            size_t            beg  = 0;
            size_t            next = std::string::npos;
            const std::string tok("\\");
            do {
                next = s.find_first_of(tok, beg);
                res.push_back(std::stof(s.substr(beg, next)));
                beg = next + tok.size();
            } while (next != std::string::npos);
        }
    }

    /// Get DICOM values in a integer vector
    /// \param bv pointer to ByteValue
    /// \param vr DICOM Value Representation: http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html
    /// \param vm DICOM Value Multiplicity: http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.4.html
    /// \param vl Value length
    /// \param res reference to a "int" container for dicom values.
    /// \return void
    void
    get_values(const gdcm::ByteValue* bv,
               const gdcm::VR         vr,
               const gdcm::VM         vm,
               const gdcm::VL         vl,
               std::vector<int>&      res,
               bool                   show = false) const {
        //works for VR of FL, DS, ...
        res.clear();

        size_t ndim = vm.GetLength();

        if (ndim == 0) {
            if (vr & gdcm::VR::FL) {       //
                ndim = vl / sizeof(int);   //T should be given 'float'
            }
        }
        assert(ndim >= 1);   //From here, dim shouldn't be 0.a

        if (vr & gdcm::VR::VRBINARY) {
            res.resize(ndim);
            bv->GetBuffer((char*) &res[0], ndim * sizeof(int));
        } else if (vr & gdcm::VR::VRASCII) {
            std::string s = std::string(bv->GetPointer(), bv->GetLength());

            size_t            beg  = 0;
            size_t            next = std::string::npos;
            const std::string tok("\\");
            do {
                next = s.find_first_of(tok, beg);
                res.push_back(std::stoi(s.substr(beg, next)));
                beg = next + tok.size();
            } while (next != std::string::npos);
        }
    }

    /// Get DICOM values in a string vector
    /// \param bv pointer to ByteValue
    /// \param vr DICOM Value Representation: http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html
    /// \param vm DICOM Value Multiplicity: http://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.4.html
    /// \param vl Value length
    /// \param res reference to a "string" container for dicom values.
    /// \return void
    void
    get_values(const gdcm::ByteValue*    bv,
               const gdcm::VR            vr,
               const gdcm::VM            vm,
               const gdcm::VL            vl,
               std::vector<std::string>& res) const {
        //works for VR of FL, DS, ...
        res.clear();

        size_t ndim = vm.GetLength();

        assert(ndim >= 1);   //From here, dim shouldn't be 0.

        if (vr & gdcm::VR::VRASCII) {
            std::string s = std::string(bv->GetPointer(), bv->GetLength());

            size_t            beg  = 0;
            size_t            next = std::string::npos;
            const std::string tok("\\");
            do {
                next = s.find_first_of(tok, beg);
                res.push_back(s.substr(beg, next));
                beg = next + tok.size();
            } while (next != std::string::npos);
        }
    }

    /// A template method to get DICOM values in a 'T' type vector
    /// \param keywoard Tag string
    /// \param result reference to a "T" container for dicom values.
    /// \tparam T type of contents.
    /// \return void
    template<typename T>
    void
    get_values(const char* keyword, std::vector<T>& result) const {
        result.clear();
        const gdcm::DataElement& el = (*this)[keyword];
        if (el.IsEmpty()) {
            result.resize(0);
            return;
        }
        const gdcm::Dicts&     dicts = gdcm::Global::GetInstance().GetDicts();
        const gdcm::Tag&       tag   = el.GetTag();
        const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);

        this->get_values(el.GetByteValue(), entry.GetVR(), entry.GetVM(), el.GetVL(), result);
    }

    /// A template method to get gdcm::DataElement in a 'T' type vector
    /// \param el DataElement
    /// \param result reference to a "T" container for dicom values.
    /// \tparam T type of contents.
    /// \return void
    template<typename T>
    void
    get_values(const gdcm::DataElement& el, std::vector<T>& result) const {
        result.clear();
        if (el.IsEmpty()) {
            result.resize(0);
            return;
        }
        const gdcm::Dicts&     dicts = gdcm::Global::GetInstance().GetDicts();
        const gdcm::Tag&       tag   = el.GetTag();
        const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);

        this->get_values(el.GetByteValue(), entry.GetVR(), entry.GetVM(), el.GetVL(), result);
    }

    /// Prints out DataElement's VR, VM, and VL.
    void
    print_dataelement(const gdcm::DataElement& el) const {
        const gdcm::Dicts& dicts = gdcm::Global::GetInstance().GetDicts();

        //convert ByteValue by using GetVR/VM
        const gdcm::ByteValue* bv = el.GetByteValue();

        const gdcm::Tag        tag   = el.GetTag();
        const gdcm::DictEntry& entry = dicts.GetDictEntry(tag);
        //el.GetType()
        //VM1-n doesn't have a number
        std::cout << "---- " << entry.GetKeyword() << " VM:" << entry.GetVM().GetLength()
                  << " VR:" << entry.GetVR() << ", GetVL: " << el.GetVL()
                  << ", GetByteValues: " << el.GetByteValue() << ", VR length"
                  << el.GetVR().GetLength() << ", VR size"
                  << el.GetVR().GetVRString(
                       el.GetVR())   //don't call (.GetSize of VR) this. it produced error
                  << ",    value:";
        if (el.IsEmpty()) {
            std::cout << "no value." << std::endl;
        } else {
            if (entry.GetVR() & gdcm::VR::FL) {

                const size_t ndim = el.GetVL() / sizeof(float);

                float c[ndim];
                bv->GetBuffer((char*) c, sizeof(c));

                std::cout << "FL encoding: ";
                for (size_t i = 0; i < ndim; ++i) {
                    std::cout << "   " << i << ": " << c[i];
                }
                std::cout << std::endl;

            } else {
            }
        }
    }
};

}   // namespace mqi
#endif
