// Scorer for MatInfo
//
// ********************************************************************
// *                                                                  *
// * This  code implementation is an extension to developments by the *
// * TOPAS collaboration.                                             *
// * This extension  is an freely  available in  accordance  with the *
// * freeBSD license:											      *
// * Copyright (c) <2015>, <Harald Paganetti>                         *
// * All rights reserved.                                             *
// * Redistribution    and   use in   source and   binary    forms,   *
// * with or without modification, are permitted provided that the    *
// * following conditions are met:                                    *
// *                                                                  *
// *                                                                  *
// * 1. Redistributions of source code must retain the above          *
// * copyright notice, this                                           *
// * list of conditions and the following disclaimer.                 *
// * 2. Redistributions in binary form must reproduce the above       *
// * copyright notice, this list of conditions and the following      *
// * disclaimer in the documentation and/or other materials provided  *
// * with the distribution.                                           *
// *                                                                  *
// * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND           *
// * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,      *
// * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF         *
// * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE         *
// * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR             *
// * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,     *
// * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT *
// * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF *
// * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED  *
// * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      *
// * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING   *
// * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF   *
// * THE POSSIBILITY OF SUCH DAMAGE.                                  *
// *                                                                  *
// * The views and conclusions contained in the software and          *
// * documentation are those of the authors and should not be         *
// * interpreted as representing official policies, either expressed  *
// * or implied, of the FreeBSD Project.                              *
// *                                                                  *
// * Contacts: Jan Schuemann, jschuemann@mgh.harvard.edu              *
// *           Harald Paganetti, hpaganetti@mgh.harvard.edu           *
// *                                                                  *
// ********************************************************************
//
//
//  This scorer creates a csv file for callibration to the HU-conversion for each institutional CT scanner
//  To be used with Patient.txt, EachHUonce.dat, and HUtoMaterialSchneiderNoCorrection.txt
//  Uses 100 MeV protons as reference to calculate stopping power ratios

#include "TsScoreMatInfo.hh"
#include "TsParameterManager.hh"
//#include "TsMaterialManager.hh"
//#include "TsGeometryManager.hh"

#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

TsScoreMatInfo::TsScoreMatInfo(TsParameterManager *pM, TsMaterialManager *mM, TsGeometryManager *gM, TsScoringManager *scM, TsExtensionManager *eM, G4String scorerName, G4String quantity, G4String outFileName, G4bool isSubScorer)
    : TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
{

    SetUnit("Gy");

    fWater = GetMaterial("G4_WATER");
    fWater = GetMaterial("Water_75eV");

    fProton = G4Proton::ProtonDefinition();

    if (fPm->ParameterExists(GetFullParmName("OutputFileName")))
        fFileName = fPm->GetStringParameter(GetFullParmName("OutputFileName")) + ".csv";
    else
    {
        G4cout << "Topas is exiting due to missing parameter." << G4endl;
        G4cout << "Scorer " << GetName() << " needs a defined OutputFileName" << G4endl;
        exit(1);
    }

    fFile.open(fFileName, std::ios::out);
    fFlag = 0;
}

TsScoreMatInfo::~TsScoreMatInfo()
{
    fFile.close();
}

G4bool TsScoreMatInfo::ProcessHits(G4Step *aStep, G4TouchableHistory *)
{
    if (!fIsActive)
    {
        fSkippedWhileInactive++;
        return false;
    }

    if (fFlag == 0)
    {
        std::vector<G4Material *> *table = aStep->GetPreStepPoint()->GetMaterial()->GetMaterialTable();
        G4int numberOfMaterials = aStep->GetPreStepPoint()->GetMaterial()->GetNumberOfMaterials();
        G4double materialStoppingPower = 0.0;

        fFile << "Material Name,Radiation Length,Density,dEdx" << std::endl;
        for (int i = 0; i < numberOfMaterials; i++)
        {
            G4Material *material = table->at(i);
            materialStoppingPower = fEmCalculator.ComputeTotalDEDX(100 * MeV, fProton, material);
            fFile << material->GetName() << "," << material->GetRadlen() / cm << "," << material->GetDensity() * cm3 / g <<","<< materialStoppingPower / (MeV / mm) << std::endl;
        }
    }
    fFlag++;
    return false;
}
