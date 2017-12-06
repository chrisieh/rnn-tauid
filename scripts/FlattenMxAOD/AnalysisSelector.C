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
#include <cmath>

void AnalysisSelector::Begin(TTree * /*tree*/)
{
    // The Begin() function is called at the start of the query.
    // When running with PROOF Begin() is only called on the client.
    // The tree argument is deprecated (on PROOF 0 is passed).

    TString option = GetOption();

    // create output file and tree (note: this will not work for proof)
    TTree::SetMaxTreeSize(9500000000); // 9.5 GB
    fOutFile = TFile::Open(fout_name.c_str(), "RECREATE");
    fOutTree = new TTree("CollectionTree", "CollectionTree");

    set_branches();
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

    fReader.SetLocalEntry(entry);

    // auto mcEventNumber = *reader_mcEventNumber;
    // // Only uneven events for training
    // if (mcEventNumber % 2 == 1) {
    //     return kTRUE;
    // }

    // for (auto ievent = 0; ievent < reader_mcEventNumber.GetSize(); ievent++) {
    //     cout << "Event: " << ievent << endl;
    // }


    for (size_t itau = 0; itau < reader_pt.GetSize(); itau++)
    {
        auto nTracks = reader_nTracks[itau];


        // Only 1- and 3-prongs
        if (nTracks != 1 && nTracks != 3)
        {
            continue;
        }

        clear_vectors();
        fill_tau(itau);
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

void AnalysisSelector::set_branches() {
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

    b_nTracks = fOutTree->Branch(
        "TauJets.nTracks", &v_nTracks, "TauJets.nTracks/I");
    b_pt = fOutTree->Branch(
        "TauJets.pt", &v_pt, "TauJets.pt/F");
    b_eta = fOutTree->Branch(
        "TauJets.eta", &v_eta, "TauJets.eta/F");
    b_phi = fOutTree->Branch(
        "TauJets.phi", &v_phi, "TauJets.phi/F");
    b_ptJetSeed = fOutTree->Branch(
        "TauJets.ptJetSeed", &v_ptJetSeed, "TauJets.ptJetSeed/F");
    b_etaJetSeed = fOutTree->Branch(
        "TauJets.etaJetSeed", &v_etaJetSeed, "TauJets.etaJetSeed/F");
    b_phiJetSeed = fOutTree->Branch(
        "TauJets.phiJetSeed", &v_phiJetSeed, "TauJets.phiJetSeed/F");
    b_nTracksTotal = fOutTree->Branch(
        "TauJets.nTracksTotal", &v_trk_nTracksTotal, "TauJets.nTracksTotal/I");
    b_cls_nClustersTotal = fOutTree->Branch(
        "TauJets.nClustersTotal", &v_cls_nClustersTotal);


    b_mu = fOutTree->Branch(
        "TauJets.mu", &v_mu, "TauJets.mu/D");
    b_nVtxPU = fOutTree->Branch(
        "TauJets.nVtxPU", &v_nVtxPU, "TauJets.nVtxPU/I");
    b_centFrac = fOutTree->Branch(
        "TauJets.centFrac", &v_centFrac, "TauJets.centFrac/F");
    b_EMPOverTrkSysP = fOutTree->Branch(
        "TauJets.EMPOverTrkSysP", &v_EMPOverTrkSysP, "TauJets.EMPOverTrkSysP/F");
    b_innerTrkAvgDist = fOutTree->Branch(
        "TauJets.innerTrkAvgDist", &v_innerTrkAvgDist, "TauJets.innerTrkAvgDist/F");
    b_ptRatioEflowApprox = fOutTree->Branch(
        "TauJets.ptRatioEflowApprox", &v_ptRatioEflowApprox, "TauJets.ptRatioEflowApprox/F");
    b_dRmax = fOutTree->Branch(
        "TauJets.dRmax", &v_dRmax, "TauJets.dRmax/F");
    b_trFlightPathSig = fOutTree->Branch(
        "TauJets.trFlightPathSig", &v_trFlightPathSig, "TauJets.trFlightPathSig/F");
    b_mEflowApprox = fOutTree->Branch(
        "TauJets.mEflowApprox", &v_mEflowApprox, "TauJets.mEflowApprox/F");
    b_SumPtTrkFrac = fOutTree->Branch(
        "TauJets.SumPtTrkFrac", &v_SumPtTrkFrac, "TauJets.SumPtTrkFrac/F");
    b_absipSigLeadTrk = fOutTree->Branch(
        "TauJets.absipSigLeadTrk", &v_absipSigLeadTrk, "TauJets.absipSigLeadTrk/F");
    b_massTrkSys = fOutTree->Branch(
        "TauJets.massTrkSys", &v_massTrkSys, "TauJets.massTrkSys/F");
    b_etOverPtLeadTrk = fOutTree->Branch(
        "TauJets.etOverPtLeadTrk", &v_etOverPtLeadTrk, "TauJets.etOverPtLeadTrk/F");
    b_ptIntermediateAxis = fOutTree->Branch(
        "TauJets.ptIntermediateAxis", &v_ptIntermediateAxis, "TauJets.ptIntermediateAxis/F");


    b_trk_pt = fOutTree->Branch(
        "TauTracks.pt", &v_trk_pt);
    b_trk_eta = fOutTree->Branch(
        "TauTracks.eta", &v_trk_eta);
    b_trk_phi = fOutTree->Branch(
        "TauTracks.phi", &v_trk_phi);
    b_trk_z0sinThetaTJVA = fOutTree->Branch(
        "TauTracks.z0sinThetaTJVA", &v_trk_z0sinThetaTJVA);
    b_trk_d0 = fOutTree->Branch(
        "TauTracks.d0", &v_trk_d0);
    b_trk_dRJetSeedAxis = fOutTree->Branch(
        "TauTracks.dRJetSeedAxis", &v_trk_dRJetSeedAxis);
    b_trk_eProbabilityHT = fOutTree->Branch(
        "TauTracks.eProbabilityHT", &v_trk_eProbabilityHT);
    b_trk_d0sig = fOutTree->Branch(
        "TauTracks.d0sig", &v_trk_d0sig);
    b_trk_nInnermostPixelHits = fOutTree->Branch(
        "TauTracks.nInnermostPixelHits", &v_trk_nInnermostPixelHits);
    b_trk_nPixelHits = fOutTree->Branch(
        "TauTracks.nPixelHits", &v_trk_nPixelHits);
    b_trk_nSCTHits = fOutTree->Branch(
        "TauTracks.nSCTHits", &v_trk_nSCTHits);


    b_cls_e = fOutTree->Branch(
        "TauClusters.e", &v_cls_e);
    b_cls_et = fOutTree->Branch(
        "TauClusters.et", &v_cls_et);
    b_cls_eta = fOutTree->Branch(
        "TauClusters.eta", &v_cls_eta);
    b_cls_phi = fOutTree->Branch(
        "TauClusters.phi", &v_cls_phi);
    b_cls_dRJetSeedAxis = fOutTree->Branch(
        "TauClusters.dRJetSeedAxis", &v_cls_dRJetSeedAxis);
    b_cls_SECOND_R = fOutTree->Branch(
        "TauClusters.SECOND_R", &v_cls_SECOND_R);
    b_cls_SECOND_LAMBDA = fOutTree->Branch(
        "TauClusters.SECOND_LAMBDA", &v_cls_SECOND_LAMBDA);
    b_cls_CENTER_LAMBDA = fOutTree->Branch(
        "TauClusters.CENTER_LAMBDA", &v_cls_CENTER_LAMBDA);
}

void AnalysisSelector::fill_tau(size_t itau) {
    // set tau properties for branches
    if (deco_truth) {
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
    v_ptJetSeed = reader_ptJetSeed[itau];
    v_etaJetSeed = reader_etaJetSeed[itau];
    v_phiJetSeed = reader_phiJetSeed[itau];
    v_trk_nTracksTotal = reader_trk_nTracksTotal[itau];
    v_cls_nClustersTotal = reader_cls_nClustersTotal[itau];

    v_mu = reader_mu[itau];
    v_nVtxPU = reader_nVtxPU[itau];
    v_centFrac = reader_centFrac[itau];
    v_EMPOverTrkSysP = reader_EMPOverTrkSysP[itau];
    v_innerTrkAvgDist = reader_innerTrkAvgDist[itau];
    v_ptRatioEflowApprox = reader_ptRatioEflowApprox[itau];
    v_dRmax = reader_dRmax[itau];
    v_trFlightPathSig = reader_trFlightPathSig[itau];
    v_mEflowApprox = reader_mEflowApprox[itau];
    v_SumPtTrkFrac = reader_SumPtTrkFrac[itau];
    v_absipSigLeadTrk = std::abs(reader_ipSigLeadTrk[itau]);
    v_massTrkSys = reader_massTrkSys[itau];
    v_etOverPtLeadTrk = reader_etOverPtLeadTrk[itau];
    v_ptIntermediateAxis = reader_ptIntermediateAxis[itau];

    // set track properties for branches
    for (size_t itrack = 0; itrack < reader_trk_pt[itau].size(); itrack++)
    {
        v_trk_pt.push_back(reader_trk_pt[itau][itrack]);
        v_trk_eta.push_back(reader_trk_eta[itau][itrack]);
        v_trk_phi.push_back(reader_trk_phi[itau][itrack]);
        v_trk_z0sinThetaTJVA.push_back(reader_trk_z0sinThetaTJVA[itau][itrack]);
        v_trk_d0.push_back(reader_trk_d0[itau][itrack]);
        v_trk_dRJetSeedAxis.push_back(reader_trk_dRJetSeedAxis[itau][itrack]);
        v_trk_eProbabilityHT.push_back(reader_trk_eProbabilityHT[itau][itrack]);
        v_trk_d0sig.push_back(reader_trk_d0sig[itau][itrack]);
        v_trk_nInnermostPixelHits.push_back(reader_trk_nInnermostPixelHits[itau][itrack]);
        v_trk_nPixelHits.push_back(reader_trk_nPixelHits[itau][itrack]);
        v_trk_nSCTHits.push_back(reader_trk_nSCTHits[itau][itrack]);
        v_trk_isLoose.push_back(reader_trk_isLoose[itau][itrack]);
        v_trk_passVertexCut.push_back(reader_trk_passVertexCut[itau][itrack]);
    }

    for (size_t icluster = 0; icluster < reader_cls_et[itau].size(); icluster++)
    {
        v_cls_e.push_back(reader_cls_e[itau][icluster]);
        v_cls_et.push_back(reader_cls_et[itau][icluster]);
        v_cls_eta.push_back(reader_cls_eta[itau][icluster]);
        v_cls_phi.push_back(reader_cls_phi[itau][icluster]);
        v_cls_dRJetSeedAxis.push_back(reader_cls_dRJetSeedAxis[itau][icluster]);
        v_cls_SECOND_R.push_back(reader_cls_SECOND_R[itau][icluster]);
        v_cls_SECOND_LAMBDA.push_back(reader_cls_SECOND_LAMBDA[itau][icluster]);
        v_cls_CENTER_LAMBDA.push_back(reader_cls_CENTER_LAMBDA[itau][icluster]);
    }
}

void AnalysisSelector::clear_vectors() {
    // clear track vectors
    v_trk_pt.clear();
    v_trk_eta.clear();
    v_trk_phi.clear();
    v_trk_z0sinThetaTJVA.clear();
    v_trk_d0.clear();
    v_trk_dRJetSeedAxis.clear();
    v_trk_eProbabilityHT.clear();
    v_trk_d0sig.clear();
    v_trk_nInnermostPixelHits.clear();
    v_trk_nPixelHits.clear();
    v_trk_nSCTHits.clear();

    // Clear cluster vectors
    v_cls_e.clear();
    v_cls_et.clear();
    v_cls_eta.clear();
    v_cls_phi.clear();
    v_cls_dRJetSeedAxis.clear();
    v_cls_SECOND_R.clear();
    v_cls_SECOND_LAMBDA.clear();
    v_cls_CENTER_LAMBDA.clear();
}
