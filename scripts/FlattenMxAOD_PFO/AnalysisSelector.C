#define AnalysisSelector_cxx
// The class definition in AnalysisSelector.h has been generated automatically
// by the ROOT utility TTree::MakeSelector(). This class is derived
// from the ROOT class TSelector. For more information on the TSelector
// framework see $ROOTSYS/README/README.SELECTOR or the ROOT User Manual.

// The following methods are defined in this file:
//   Begin():      called every time a loop on the tree starts,
//              a convenient place to create your histograms.
//   SlaveBegin():  called after Begin(), when on PROOF called only on the
//              slave servers.
//   Process():    called for each event, in this function you decide what
//              to read and fill your histograms.
//   SlaveTerminate: called at the end of the loop on the tree, when on PROOF
//              called only on the slave servers.
//   Terminate():   called at the end of the loop on the tree,
//              a convenient place to draw/fit your histograms.
//
// To use this file, try the following session on your Tree T:
//
// root> T->Process("AnalysisSelector.C")
// root> T->Process("AnalysisSelector.C","some options")
// root> T->Process("AnalysisSelector.C+")
//

#include "AnalysisSelector.h"
#include <TH2.h>
#include <TStyle.h>
#include <cmath>

void AnalysisSelector::Begin(TTree * /*tree*/)
{
    // The Begin() function is called at the start of the query.
    // When running with PROOF Begin() is only called on the client.
    // The tree argument is deprecated (on PROOF 0 is passed).

    TString option = GetOption();

    // create output file and tree (note: this will not work for proof)
    fOutFile = TFile::Open(fout_name.c_str(), "RECREATE");
    fOutTree = new TTree("CollectionTree", "CollectionTree");

    // set branches for output tree
    if (deco_truth)
    {
        b_truthProng = fOutTree->Branch(
            "TauJets.truthProng", &v_truthProng, "TauJets.truthProng/l");
        b_truthEtaVis = fOutTree->Branch(
            "TauJets.truthEtaVis", &v_truthEtaVis, "TauJets.truthEtaVis/D");
        b_truthPtVis = fOutTree->Branch(
            "TauJets.truthPtVis", &v_truthPtVis, "TauJets.truthPtVis/D");
        b_IsTruthMatched = fOutTree->Branch(
            "TauJets.IsTruthMatched", &v_IsTruthMatched, "TauJets.IsTruthMatched/B");
        b_truthDecayMode = fOutTree->Branch(
            "TauJets.truthDecayMode", &v_truthDecayMode, "TauJets.truthDecayMode/I");
    }

    b_pt = fOutTree->Branch(
        "TauJets.pt", &v_pt, "TauJets.pt/F");
    b_eta = fOutTree->Branch(
        "TauJets.eta", &v_eta, "TauJets.eta/F");
    b_phi = fOutTree->Branch(
        "TauJets.phi", &v_phi, "TauJets.phi/F");
    b_nTracks = fOutTree->Branch(
        "TauJets.nTracks", &v_nTracks, "TauJets.nTracks/I");
    b_mu = fOutTree->Branch(
        "TauJets.mu", &v_mu, "TauJets.mu/D");
    b_nVtxPU = fOutTree->Branch(
        "TauJets.nVtxPU", &v_nVtxPU, "TauJets.nVtxPU/I");

    b_PanTau_DecayModeProto = fOutTree->Branch(
        "TauJets.PanTau_DecayModeProto", &v_PanTau_DecayModeProto, "TauJets.PanTau_DecayModeProto/I");
    b_PanTau_DecayMode = fOutTree->Branch(
        "TauJets.PanTau_DecayMode", &v_PanTau_DecayMode, "TauJets.PanTau_DecayMode/I");
    b_jet_Pt = fOutTree->Branch(
        "TauJets.Pt", &v_jet_Pt, "TauJets.Pt/F");
    b_jet_Phi = fOutTree->Branch(
        "TauJets.Phi", &v_jet_Phi, "TauJets.Phi/F");
    b_jet_Eta = fOutTree->Branch(
        "TauJets.Eta", &v_jet_Eta, "TauJets.Eta/F");
    b_jet_nChargedPFOs = fOutTree->Branch(
        "TauJets.nChargedPFOs", &v_jet_nChargedPFOs, "TauJets.nChargedPFOs/b");
    b_jet_nNeutralPFOs = fOutTree->Branch(
        "TauJets.nNeutralPFOs", &v_jet_nNeutralPFOs, "TauJets.nNeutralPFOs/b");
    b_jet_nShotPFOs = fOutTree->Branch(
        "TauJets.nShotPFOs", &v_jet_nShotPFOs, "TauJets.nShotPFOs/b");
    b_jet_nHadronicPFOs = fOutTree->Branch(
        "TauJets.nHadronicPFOs", &v_jet_nHadronicPFOs, "TauJets.nHadronicPFOs/b");
    b_jet_nConversion = fOutTree->Branch(
        "TauJets.nConversion", &v_jet_nConversion, "TauJets.nConversion/b");

    // PFOs
    b_pfo_chargedPt = fOutTree->Branch(
        "TauPFOs.chargedPt", &v_pfo_chargedPt);
    b_pfo_chargedPhi = fOutTree->Branch(
        "TauPFOs.chargedPhi", &v_pfo_chargedPhi);
    b_pfo_chargedEta = fOutTree->Branch(
        "TauPFOs.chargedEta", &v_pfo_chargedEta);

    b_pfo_neutralPt = fOutTree->Branch(
        "TauPFOs.neutralPt", &v_pfo_neutralPt);
    b_pfo_neutralPhi = fOutTree->Branch(
        "TauPFOs.neutralPhi", &v_pfo_neutralPhi);
    b_pfo_neutralEta = fOutTree->Branch(
        "TauPFOs.neutralEta", &v_pfo_neutralEta);
    b_pfo_neutralPi0BDT = fOutTree->Branch(
        "TauPFOs.neutralPi0BDT", &v_pfo_neutralPi0BDT);
    b_pfo_neutralNHitsInEM1 = fOutTree->Branch(
        "TauPFOs.neutralNHitsInEM1", &v_pfo_neutralNHitsInEM1);

    b_pfo_neutralPtSub = fOutTree->Branch(
        "TauPFOs.neutralPtSub", &v_pfo_neutralPtSub);
    b_pfo_neutral_SECOND_R = fOutTree->Branch(
        "TauPFOs.neutral_SECOND_R", &v_pfo_neutral_SECOND_R);
    b_pfo_neutral_SECOND_LAMBDA = fOutTree->Branch(
        "TauPFOs.neutral_SECOND_LAMBDA", &v_pfo_neutral_SECOND_LAMBDA);
    b_pfo_neutral_CENTER_LAMBDA = fOutTree->Branch(
        "TauPFOs.neutral_CENTER_LAMBDA", &v_pfo_neutral_CENTER_LAMBDA);
    b_pfo_neutral_ENG_FRAC_MAX = fOutTree->Branch(
        "TauPFOs.neutral_ENG_FRAC_MAX", &v_pfo_neutral_ENG_FRAC_MAX);
    b_pfo_neutral_ENG_FRAC_CORE = fOutTree->Branch(
        "TauPFOs.neutral_ENG_FRAC_CORE", &v_pfo_neutral_ENG_FRAC_CORE);
    b_pfo_neutral_SECOND_ENG_DENS = fOutTree->Branch(
        "TauPFOs.neutral_SECOND_ENG_DENS", &v_pfo_neutral_SECOND_ENG_DENS);
    b_pfo_neutral_NPosECells_EM1 = fOutTree->Branch(
        "TauPFOs.neutral_NPosECells_EM1", &v_pfo_neutral_NPosECells_EM1);
    b_pfo_neutral_NPosECells_EM2 = fOutTree->Branch(
        "TauPFOs.neutral_NPosECells_EM2", &v_pfo_neutral_NPosECells_EM2);
    b_pfo_neutral_secondEtaWRTClusterPosition_EM1 = fOutTree->Branch(
        "TauPFOs.neutral_secondEtaWRTClusterPosition_EM1",
        &v_pfo_neutral_secondEtaWRTClusterPosition_EM1);
    b_pfo_neutral_secondEtaWRTClusterPosition_EM2 = fOutTree->Branch(
        "TauPFOs.neutral_secondEtaWRTClusterPosition_EM2",
        &v_pfo_neutral_secondEtaWRTClusterPosition_EM2);
    b_pfo_neutral_energyfrac_EM1 = fOutTree->Branch(
        "TauPFOs.neutral_energyfrac_EM1", &v_pfo_neutral_energyfrac_EM1);
    b_pfo_neutral_energyfrac_EM2 = fOutTree->Branch(
        "TauPFOs.neutral_energyfrac_EM2", &v_pfo_neutral_energyfrac_EM2);

    b_pfo_neutralPt_BDTSort = fOutTree->Branch(
        "TauPFOs.neutralPt_BDTSort", &v_pfo_neutralPt_BDTSort);
    b_pfo_neutralPhi_BDTSort = fOutTree->Branch(
        "TauPFOs.neutralPhi_BDTSort", &v_pfo_neutralPhi_BDTSort);
    b_pfo_neutralEta_BDTSort = fOutTree->Branch(
        "TauPFOs.neutralEta_BDTSort", &v_pfo_neutralEta_BDTSort);
    b_pfo_neutralPi0BDT_BDTSort = fOutTree->Branch(
        "TauPFOs.neutralPi0BDT_BDTSort", &v_pfo_neutralPi0BDT_BDTSort);
    b_pfo_neutralNHitsInEM1_BDTSort = fOutTree->Branch(
        "TauPFOs.neutralNHitsInEM1_BDTSort", &v_pfo_neutralNHitsInEM1_BDTSort);

    b_pfo_shotPt = fOutTree->Branch(
        "TauPFOs.shotPt", &v_pfo_shotPt);
    b_pfo_shotPhi = fOutTree->Branch(
        "TauPFOs.shotPhi", &v_pfo_shotPhi);
    b_pfo_shotEta = fOutTree->Branch(
        "TauPFOs.shotEta", &v_pfo_shotEta);

    b_pfo_hadronicPt = fOutTree->Branch(
        "TauPFOs.hadronicPt", &v_pfo_hadronicPt);
    b_pfo_hadronicPhi = fOutTree->Branch(
        "TauPFOs.hadronicPhi", &v_pfo_hadronicPhi);
    b_pfo_hadronicEta = fOutTree->Branch(
        "TauPFOs.hadronicEta", &v_pfo_hadronicEta);

    b_conv_pt = fOutTree->Branch(
        "TauConv.pt", &v_conv_pt);
    b_conv_phi = fOutTree->Branch(
        "TauConv.phi", &v_conv_phi);
    b_conv_eta = fOutTree->Branch(
        "TauConv.eta", &v_conv_eta);
    b_conv_phi_extrap = fOutTree->Branch(
        "TauConv.phi_extrap", &v_conv_phi_extrap);
    b_conv_eta_extrap = fOutTree->Branch(
        "TauConv.eta_extrap", &v_conv_eta_extrap);
}

void AnalysisSelector::SlaveBegin(TTree * /*tree*/)
{
    // The SlaveBegin() function is called after the Begin() function.
    // When running with PROOF SlaveBegin() is called on each slave server.
    // The tree argument is deprecated (on PROOF 0 is passed).

    TString option = GetOption();
}

Bool_t AnalysisSelector::Process(Long64_t entry)
{
    // The Process() function is called for each entry in the tree (or possibly
    // keyed object in the case of PROOF) to be processed. The entry argument
    // specifies which entry in the currently loaded tree is to be processed.
    // When processing keyed objects with PROOF, the object is already loaded
    // and is available via the fObject pointer.
    //
    // This function should contain the \"body\" of the analysis. It can contain
    // simple or elaborate selection criteria, run algorithms on the data
    // of the event and typically fill histograms.
    //
    // The processing can be stopped by calling Abort().
    //
    // Use fStatus to set the return value of TTree::Process().
    //
    // The return value is currently not used.

    fReader.SetEntry(entry);

    for (size_t itau = 0; itau < reader_pt.GetSize(); itau++)
    {
        // Clear PFO vectors
        v_pfo_chargedPt.clear();
        v_pfo_chargedPhi.clear();
        v_pfo_chargedEta.clear();

        v_pfo_neutralPt.clear();
        v_pfo_neutralPhi.clear();
        v_pfo_neutralEta.clear();
        v_pfo_neutralPi0BDT.clear();
        v_pfo_neutralNHitsInEM1.clear();

        v_pfo_neutralPtSub.clear();
        v_pfo_neutral_SECOND_R.clear();
        v_pfo_neutral_SECOND_LAMBDA.clear();
        v_pfo_neutral_CENTER_LAMBDA.clear();
        v_pfo_neutral_ENG_FRAC_MAX.clear();
        v_pfo_neutral_ENG_FRAC_CORE.clear();
        v_pfo_neutral_SECOND_ENG_DENS.clear();
        v_pfo_neutral_NPosECells_EM1.clear();
        v_pfo_neutral_NPosECells_EM2.clear();
        v_pfo_neutral_secondEtaWRTClusterPosition_EM1.clear();
        v_pfo_neutral_secondEtaWRTClusterPosition_EM2.clear();
        v_pfo_neutral_energyfrac_EM1.clear();
        v_pfo_neutral_energyfrac_EM2.clear();

        v_pfo_neutralPt_BDTSort.clear();
        v_pfo_neutralPhi_BDTSort.clear();
        v_pfo_neutralEta_BDTSort.clear();
        v_pfo_neutralPi0BDT_BDTSort.clear();
        v_pfo_neutralNHitsInEM1_BDTSort.clear();

        v_pfo_shotPt.clear();
        v_pfo_shotPhi.clear();
        v_pfo_shotEta.clear();

        v_pfo_hadronicPt.clear();
        v_pfo_hadronicPhi.clear();
        v_pfo_hadronicEta.clear();

        v_conv_pt.clear();
        v_conv_phi.clear();
        v_conv_eta.clear();
        v_conv_phi_extrap.clear();
        v_conv_eta_extrap.clear();

        // set tau properties for branches
        if (deco_truth)
        {
            v_truthProng = (*reader_truthProng)[itau];
            v_truthEtaVis = (*reader_truthEtaVis)[itau];
            v_truthPtVis = (*reader_truthPtVis)[itau];
            v_IsTruthMatched = (*reader_IsTruthMatched)[itau];
            v_truthDecayMode = (*reader_truthDecayMode)[itau];
        }

        // TauJet variables
        v_pt = reader_pt[itau];
        v_eta = reader_eta[itau];
        v_phi = reader_phi[itau];
        v_nTracks = reader_nTracks[itau];
        v_mu = reader_mu[itau];
        v_nVtxPU = reader_nVtxPU[itau];
        v_PanTau_DecayModeProto = reader_PanTau_DecayModeProto[itau];
        v_PanTau_DecayMode = reader_PanTau_DecayMode[itau];

        v_jet_Pt = reader_jet_Pt[itau];
        v_jet_Phi = reader_jet_Phi[itau];
        v_jet_Eta = reader_jet_Eta[itau];
        v_jet_nChargedPFOs = reader_jet_nChargedPFOs[itau];
        v_jet_nNeutralPFOs = reader_jet_nNeutralPFOs[itau];
        v_jet_nShotPFOs = reader_jet_nShotPFOs[itau];
        v_jet_nHadronicPFOs = reader_jet_nHadronicPFOs[itau];
        v_jet_nConversion = reader_jet_nConversion[itau];

        // Charged PFOs
        for (size_t iPFO = 0; iPFO < reader_pfo_chargedPt[itau].size(); iPFO++)
        {
            v_pfo_chargedPt.push_back(reader_pfo_chargedPt[itau][iPFO]);
            v_pfo_chargedPhi.push_back(reader_pfo_chargedPhi[itau][iPFO]);
            v_pfo_chargedEta.push_back(reader_pfo_chargedEta[itau][iPFO]);
        }

        // Neutral PFOs
        for (size_t iPFO = 0; iPFO < reader_pfo_neutralPt[itau].size(); iPFO++)
        {
            v_pfo_neutralPt.push_back(reader_pfo_neutralPt[itau][iPFO]);
            v_pfo_neutralPhi.push_back(reader_pfo_neutralPhi[itau][iPFO]);
            v_pfo_neutralEta.push_back(reader_pfo_neutralEta[itau][iPFO]);
            v_pfo_neutralPi0BDT.push_back(reader_pfo_neutralPi0BDT[itau][iPFO]);
            v_pfo_neutralNHitsInEM1.push_back(reader_pfo_neutralNHitsInEM1[itau][iPFO]);

            v_pfo_neutralPtSub.push_back(
                reader_pfo_neutralPtSub[itau][iPFO]);
            v_pfo_neutral_SECOND_R.push_back(
                reader_pfo_neutral_SECOND_R[itau][iPFO]);
            v_pfo_neutral_SECOND_LAMBDA.push_back(
                reader_pfo_neutral_SECOND_LAMBDA[itau][iPFO]);
            v_pfo_neutral_CENTER_LAMBDA.push_back(
                reader_pfo_neutral_CENTER_LAMBDA[itau][iPFO]);
            v_pfo_neutral_ENG_FRAC_MAX.push_back(
                reader_pfo_neutral_ENG_FRAC_MAX[itau][iPFO]);
            v_pfo_neutral_ENG_FRAC_CORE.push_back(
                reader_pfo_neutral_ENG_FRAC_CORE[itau][iPFO]);
            v_pfo_neutral_SECOND_ENG_DENS.push_back(
                reader_pfo_neutral_SECOND_ENG_DENS[itau][iPFO]);
            v_pfo_neutral_NPosECells_EM1.push_back(
                reader_pfo_neutral_NPosECells_EM1[itau][iPFO]);
            v_pfo_neutral_NPosECells_EM2.push_back(
                reader_pfo_neutral_NPosECells_EM2[itau][iPFO]);
            v_pfo_neutral_secondEtaWRTClusterPosition_EM1.push_back(
                reader_pfo_neutral_secondEtaWRTClusterPosition_EM1[itau][iPFO]);
            v_pfo_neutral_secondEtaWRTClusterPosition_EM2.push_back(
                reader_pfo_neutral_secondEtaWRTClusterPosition_EM2[itau][iPFO]);
            v_pfo_neutral_energyfrac_EM1.push_back(
                reader_pfo_neutral_energyfrac_EM1[itau][iPFO]);
            v_pfo_neutral_energyfrac_EM2.push_back(
                reader_pfo_neutral_energyfrac_EM2[itau][iPFO]);
        }

        // Neutral PFOs (BDTSort)
        for (size_t iPFO = 0; iPFO < reader_pfo_neutralPt_BDTSort[itau].size(); iPFO++) {
            v_pfo_neutralPt_BDTSort.push_back(
                reader_pfo_neutralPt_BDTSort[itau][iPFO]);
            v_pfo_neutralPhi_BDTSort.push_back(
                reader_pfo_neutralPhi_BDTSort[itau][iPFO]);
            v_pfo_neutralEta_BDTSort.push_back(
                reader_pfo_neutralEta_BDTSort[itau][iPFO]);
            v_pfo_neutralPi0BDT_BDTSort.push_back(
                reader_pfo_neutralPi0BDT_BDTSort[itau][iPFO]);
            v_pfo_neutralNHitsInEM1_BDTSort.push_back(
                reader_pfo_neutralNHitsInEM1_BDTSort[itau][iPFO]);
        }

        // Shots
        for (size_t iPFO = 0; iPFO < reader_pfo_shotPt[itau].size(); iPFO++) {
            v_pfo_shotPt.push_back(
                reader_pfo_shotPt[itau][iPFO]);
            v_pfo_shotPhi.push_back(
                reader_pfo_shotPhi[itau][iPFO]);
            v_pfo_shotEta.push_back(
                reader_pfo_shotEta[itau][iPFO]);
        }

        // Hadronic PFOs
        for (size_t iPFO = 0; iPFO < reader_pfo_hadronicPt[itau].size(); iPFO++) {
            v_pfo_hadronicPt.push_back(
                reader_pfo_hadronicPt[itau][iPFO]);
            v_pfo_hadronicPhi.push_back(
                reader_pfo_hadronicPhi[itau][iPFO]);
            v_pfo_hadronicEta.push_back(
                reader_pfo_hadronicEta[itau][iPFO]);
        }

        // Conversion tracks
        for (size_t iTrack = 0; iTrack < reader_conv_pt[itau].size(); iTrack++) {
            v_conv_pt.push_back(
                reader_conv_pt[itau][iTrack]);
            v_conv_phi.push_back(
                reader_conv_phi[itau][iTrack]);
            v_conv_eta.push_back(
                reader_conv_eta[itau][iTrack]);
            v_conv_phi_extrap.push_back(
                reader_conv_phi_extrap[itau][iTrack]);
            v_conv_eta_extrap.push_back(
                reader_conv_eta_extrap[itau][iTrack]);
        }

        fOutTree->Fill();
    }

    return kTRUE;
}

void AnalysisSelector::SlaveTerminate()
{
    // The SlaveTerminate() function is called after all entries or objects
    // have been processed. When running with PROOF SlaveTerminate() is called
    // on each slave server.
}

void AnalysisSelector::Terminate()
{
    // The Terminate() function is the last function to be called during
    // a query. It always runs on the client, it can be used to present
    // the results graphically or save the results to file.

    fOutFile->Write();
    fOutFile->Close();
}
