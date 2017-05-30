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

    b_PanTau_DecayMode = fOutTree->Branch(
        "TauJets.PanTau_DecayMode", &v_PanTau_DecayMode, "TauJets.PanTau_DecayMode/I");
    b_jet_Px = fOutTree->Branch(
        "TauJets.Px", &v_jet_Px, "TauJets.Px/F");
    b_jet_Py = fOutTree->Branch(
        "TauJets.Py", &v_jet_Py, "TauJets.Py/F");
    b_jet_Pz = fOutTree->Branch(
        "TauJets.Pz", &v_jet_Pz, "TauJets.Pz/F");
    b_jet_E = fOutTree->Branch(
        "TauJets.E", &v_jet_E, "TauJets.E/F");
    b_jet_Phi = fOutTree->Branch(
        "TauJets.Phi", &v_jet_Phi, "TauJets.Phi/F");
    b_jet_Eta = fOutTree->Branch(
        "TauJets.Eta", &v_jet_Eta, "TauJets.Eta/F");
    b_jet_nChargedPFOs = fOutTree->Branch(
        "TauJets.nChargedPFOs", &v_jet_nChargedPFOs, "TauJets.nChargedPFOs/b");
    b_jet_nNeutralPFOs = fOutTree->Branch(
        "TauJets.nNeutralPFOs", &v_jet_nNeutralPFOs, "TauJets.nNeutralPFOs/b");

    // PFOs
    b_pfo_chargedPx = fOutTree->Branch(
        "TauPFOs.chargedPx", &v_pfo_chargedPx);
    b_pfo_chargedPy = fOutTree->Branch(
        "TauPFOs.chargedPy", &v_pfo_chargedPy);
    b_pfo_chargedPz = fOutTree->Branch(
        "TauPFOs.chargedPz", &v_pfo_chargedPz);
    b_pfo_chargedE = fOutTree->Branch(
        "TauPFOs.chargedE", &v_pfo_chargedE);
    b_pfo_chargedPhi = fOutTree->Branch(
        "TauPFOs.chargedPhi", &v_pfo_chargedPhi);
    b_pfo_chargedEta = fOutTree->Branch(
        "TauPFOs.chargedEta", &v_pfo_chargedEta);

    b_pfo_neutralPx = fOutTree->Branch(
        "TauPFOs.neutralPx", &v_pfo_neutralPx);
    b_pfo_neutralPy = fOutTree->Branch(
        "TauPFOs.neutralPy", &v_pfo_neutralPy);
    b_pfo_neutralPz = fOutTree->Branch(
        "TauPFOs.neutralPz", &v_pfo_neutralPz);
    b_pfo_neutralE = fOutTree->Branch(
        "TauPFOs.neutralE", &v_pfo_neutralE);
    b_pfo_neutralPhi = fOutTree->Branch(
        "TauPFOs.neutralPhi", &v_pfo_neutralPhi);
    b_pfo_neutralEta = fOutTree->Branch(
        "TauPFOs.neutralEta", &v_pfo_neutralEta);

    b_pfo_neutralPi0BDT = fOutTree->Branch(
        "TauPFOs.neutralPi0BDT", &v_pfo_neutralPi0BDT);
    b_pfo_neutralNHitsInEM1 = fOutTree->Branch(
        "TauPFOs.neutralNHitsInEM1", &v_pfo_neutralNHitsInEM1);
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
        //std::cout << "pt: " << reader_pt[itau] << std::endl;
        // auto nTracks = reader_nTracks[itau];
        // if (nTracks != 1 && nTracks != 3)
        // {
        //     continue;
        // }

        // Clear PFO vectors
        v_pfo_chargedPx.clear();
        v_pfo_chargedPy.clear();
        v_pfo_chargedPz.clear();
        v_pfo_chargedE.clear();
        v_pfo_chargedPhi.clear();
        v_pfo_chargedEta.clear();

        v_pfo_neutralPx.clear();
        v_pfo_neutralPy.clear();
        v_pfo_neutralPz.clear();
        v_pfo_neutralE.clear();
        v_pfo_neutralPhi.clear();
        v_pfo_neutralEta.clear();
        v_pfo_neutralPi0BDT.clear();
        v_pfo_neutralNHitsInEM1.clear();

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
        v_PanTau_DecayMode = reader_PanTau_DecayMode[itau];

        v_jet_Px = reader_jet_Px[itau];
        v_jet_Py = reader_jet_Py[itau];
        v_jet_Pz = reader_jet_Pz[itau];
        v_jet_E = reader_jet_E[itau];
        v_jet_Phi = reader_jet_Phi[itau];
        v_jet_Eta = reader_jet_Eta[itau];
        v_jet_nChargedPFOs = reader_jet_nChargedPFOs[itau];
        v_jet_nNeutralPFOs = reader_jet_nNeutralPFOs[itau];

        // Charged PFOs
        for (size_t iPFO = 0; iPFO < reader_pfo_chargedPx[itau].size(); iPFO++)
        {
            v_pfo_chargedPx.push_back(reader_pfo_chargedPx[itau][iPFO]);
            v_pfo_chargedPy.push_back(reader_pfo_chargedPy[itau][iPFO]);
            v_pfo_chargedPz.push_back(reader_pfo_chargedPz[itau][iPFO]);
            v_pfo_chargedE.push_back(reader_pfo_chargedE[itau][iPFO]);
            v_pfo_chargedPhi.push_back(reader_pfo_chargedPhi[itau][iPFO]);
            v_pfo_chargedEta.push_back(reader_pfo_chargedEta[itau][iPFO]);
        }

        // Neutral PFOs
        for (size_t iPFO = 0; iPFO < reader_pfo_neutralPx[itau].size(); iPFO++)
        {
            v_pfo_neutralPx.push_back(reader_pfo_neutralPx[itau][iPFO]);
            v_pfo_neutralPy.push_back(reader_pfo_neutralPy[itau][iPFO]);
            v_pfo_neutralPz.push_back(reader_pfo_neutralPz[itau][iPFO]);
            v_pfo_neutralE.push_back(reader_pfo_neutralE[itau][iPFO]);
            v_pfo_neutralPhi.push_back(reader_pfo_neutralPhi[itau][iPFO]);
            v_pfo_neutralEta.push_back(reader_pfo_neutralEta[itau][iPFO]);
            v_pfo_neutralPi0BDT.push_back(reader_pfo_neutralPi0BDT[itau][iPFO]);
            v_pfo_neutralNHitsInEM1.push_back(reader_pfo_neutralNHitsInEM1[itau][iPFO]);
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
