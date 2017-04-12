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
    b_ptJetSeed = fOutTree->Branch(
        "TauJets.ptJetSeed", &v_ptJetSeed, "TauJets.ptJetSeed/F");
    b_etaJetSeed = fOutTree->Branch(
        "TauJets.etaJetSeed", &v_etaJetSeed, "TauJets.etaJetSeed/F");
    b_phiJetSeed = fOutTree->Branch(
        "TauJets.phiJetSeed", &v_phiJetSeed, "TauJets.phiJetSeed/F");
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

    // nTracksTotal
    b_nTracksTotal = fOutTree->Branch(
        "TauJets.nTracksTotal", &v_trk_nTracksTotal, "TauJets.nTracksTotal/I");

    b_trk_pt = fOutTree->Branch(
        "TauTracks.pt", &v_trk_pt);
    b_trk_qOverP = fOutTree->Branch(
        "TauTracks.qOverP", &v_trk_qOverP);
    b_trk_eta = fOutTree->Branch(
        "TauTracks.eta", &v_trk_eta);
    b_trk_phi = fOutTree->Branch(
        "TauTracks.phi", &v_trk_phi);
    b_trk_eta_at_emcal = fOutTree->Branch(
        "TauTracks.eta_at_emcal", &v_trk_eta_at_emcal);
    b_trk_phi_at_emcal = fOutTree->Branch(
        "TauTracks.phi_at_emcal", &v_trk_phi_at_emcal);
    b_trk_z0sinThetaTJVA = fOutTree->Branch(
        "TauTracks.z0sinThetaTJVA", &v_trk_z0sinThetaTJVA);
    b_trk_d0 = fOutTree->Branch(
        "TauTracks.d0", &v_trk_d0);
    b_trk_dRJetSeedAxis = fOutTree->Branch(
        "TauTracks.dRJetSeedAxis", &v_trk_dRJetSeedAxis);
    b_trk_rConvII = fOutTree->Branch(
        "TauTracks.rConvII", &v_trk_rConvII);
    b_trk_nInnermostPixelHits = fOutTree->Branch(
        "TauTracks.nInnermostPixelHits", &v_trk_nInnermostPixelHits);
    b_trk_nPixelSharedHits = fOutTree->Branch(
        "TauTracks.nPixelSharedHits", &v_trk_nPixelSharedHits);
    b_trk_nSCTSharedHits = fOutTree->Branch(
        "TauTracks.nSCTSharedHits", &v_trk_nSCTSharedHits);
    b_trk_nPixelHits = fOutTree->Branch(
        "TauTracks.nPixelHits", &v_trk_nPixelHits);
    b_trk_nSiHits = fOutTree->Branch(
        "TauTracks.nSiHits", &v_trk_nSiHits);
    b_trk_eProbabilityHT = fOutTree->Branch(
        "TauTracks.eProbabilityHT", &v_trk_eProbabilityHT);
    b_trk_trackClassification = fOutTree->Branch(
        "TauTracks.trackClassification", &v_trk_trackClassification);
    b_trk_isCharged = fOutTree->Branch(
        "TauTracks.isCharged", &v_trk_isCharged);
    b_trk_isConversion = fOutTree->Branch(
        "TauTracks.isConversion", &v_trk_isConversion);
    b_trk_isFakeOrIsolation = fOutTree->Branch(
        "TauTracks.isFakeOrIsolation", &v_trk_isFakeOrIsolation);

    b_trk_nPixelHoles = fOutTree->Branch(
        "TauTracks.nPixelHoles", &v_trk_nPixelHoles);
    b_trk_nPixelDead = fOutTree->Branch(
        "TauTracks.nPixelDead", &v_trk_nPixelDead);
    b_trk_nSCTHits = fOutTree->Branch(
        "TauTracks.nSCTHits", &v_trk_nSCTHits);
    b_trk_nSCTHoles = fOutTree->Branch(
        "TauTracks.nSCTHoles", &v_trk_nSCTHoles);
    b_trk_nSCTDead = fOutTree->Branch(
        "TauTracks.nSCTDead", &v_trk_nSCTDead);
    b_trk_nSCTDoubleHoles = fOutTree->Branch(
        "TauTracks.nSCTDoubleHoles", &v_trk_nSCTDoubleHoles);
    b_trk_nTRTHits = fOutTree->Branch(
        "TauTracks.nTRTHits", &v_trk_nTRTHits);
    b_trk_nTRTDead = fOutTree->Branch(
        "TauTracks.nTRTDead", &v_trk_nTRTDead);

    // Set cluster branches
    b_cls_nClustersTotal = fOutTree->Branch("TauJets.nClustersTotal", &v_cls_nClustersTotal);
    b_cls_e = fOutTree->Branch("TauClusters.e", &v_cls_e);
    b_cls_et = fOutTree->Branch("TauClusters.et", &v_cls_et);
    b_cls_eta = fOutTree->Branch("TauClusters.eta", &v_cls_eta);
    b_cls_phi = fOutTree->Branch("TauClusters.phi", &v_cls_phi);
    b_cls_psfrac = fOutTree->Branch("TauClusters.psfrac", &v_cls_psfrac);
    b_cls_em1frac = fOutTree->Branch("TauClusters.em1frac", &v_cls_em1frac);
    b_cls_em2frac = fOutTree->Branch("TauClusters.em2frac", &v_cls_em2frac);
    b_cls_em3frac = fOutTree->Branch("TauClusters.em3frac", &v_cls_em3frac);
    b_cls_dRJetSeedAxis = fOutTree->Branch("TauClusters.dRJetSeedAxis", &v_cls_dRJetSeedAxis);
    b_cls_EM_PROBABILITY = fOutTree->Branch("TauClusters.EM_PROBABILITY", &v_cls_EM_PROBABILITY);
    b_cls_SECOND_R = fOutTree->Branch("TauClusters.SECOND_R", &v_cls_SECOND_R);
    b_cls_SECOND_LAMBDA = fOutTree->Branch("TauClusters.SECOND_LAMBDA", &v_cls_SECOND_LAMBDA);
    b_cls_FIRST_ENG_DENS = fOutTree->Branch("TauClusters.FIRST_ENG_DENS", &v_cls_FIRST_ENG_DENS);
    b_cls_CENTER_LAMBDA = fOutTree->Branch("TauClusters.CENTER_LAMBDA", &v_cls_CENTER_LAMBDA);
    b_cls_ENG_FRAC_MAX = fOutTree->Branch("TauClusters.ENG_FRAC_MAX", &v_cls_ENG_FRAC_MAX);
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
        auto nTracks = reader_nTracks[itau];
        if (nTracks != 1 && nTracks != 3)
        {
            continue;
        }

        // clear track vectors
        v_trk_pt.clear();
        v_trk_qOverP.clear();
        v_trk_eta.clear();
        v_trk_phi.clear();
        v_trk_eta_at_emcal.clear();
        v_trk_phi_at_emcal.clear();
        v_trk_z0sinThetaTJVA.clear();
        v_trk_d0.clear();
        v_trk_dRJetSeedAxis.clear();
        v_trk_rConvII.clear();
        v_trk_nInnermostPixelHits.clear();
        v_trk_nPixelSharedHits.clear();
        v_trk_nSCTSharedHits.clear();
        v_trk_nPixelHits.clear();
        v_trk_nSiHits.clear();
        v_trk_eProbabilityHT.clear();
        v_trk_trackClassification.clear();
        v_trk_isCharged.clear();
        v_trk_isConversion.clear();
        v_trk_isFakeOrIsolation.clear();

        v_trk_nPixelHoles.clear();
        v_trk_nPixelDead.clear();
        v_trk_nSCTHits.clear();
        v_trk_nSCTHoles.clear();
        v_trk_nSCTDead.clear();
        v_trk_nSCTDoubleHoles.clear();
        v_trk_nTRTHits.clear();
        v_trk_nTRTDead.clear();
        
        // Clear cluster vectors
        v_cls_e.clear();
        v_cls_et.clear();
        v_cls_eta.clear();
        v_cls_phi.clear();
        v_cls_psfrac.clear();
        v_cls_em1frac.clear();
        v_cls_em2frac.clear();
        v_cls_em3frac.clear();
        v_cls_dRJetSeedAxis.clear();
        v_cls_EM_PROBABILITY.clear();
        v_cls_SECOND_R.clear();
        v_cls_SECOND_LAMBDA.clear();
        v_cls_FIRST_ENG_DENS.clear();
        v_cls_CENTER_LAMBDA.clear();
        v_cls_ENG_FRAC_MAX.clear();
        
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
        v_ptJetSeed = reader_ptJetSeed[itau];
        v_etaJetSeed = reader_etaJetSeed[itau];
        v_phiJetSeed = reader_phiJetSeed[itau];
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

        // nTracksTotal
        v_trk_nTracksTotal = reader_trk_nTracksTotal[itau];
        
        // nClustersTotal
        v_cls_nClustersTotal = reader_cls_nClustersTotal[itau];

        // set track properties for branches
        for (size_t itrack = 0; itrack < reader_trk_qOverP[itau].size(); itrack++)
        {
            v_trk_pt.push_back(reader_trk_pt[itau][itrack]);
            v_trk_qOverP.push_back(reader_trk_qOverP[itau][itrack]);
            v_trk_eta.push_back(reader_trk_eta[itau][itrack]);
            v_trk_phi.push_back(reader_trk_phi[itau][itrack]);
            v_trk_eta_at_emcal.push_back(reader_trk_eta_at_emcal[itau][itrack]);
            v_trk_phi_at_emcal.push_back(reader_trk_phi_at_emcal[itau][itrack]);
            v_trk_z0sinThetaTJVA.push_back(reader_trk_z0sinThetaTJVA[itau][itrack]);
            v_trk_d0.push_back(reader_trk_d0[itau][itrack]);
            v_trk_dRJetSeedAxis.push_back(reader_trk_dRJetSeedAxis[itau][itrack]);
            v_trk_rConvII.push_back(reader_trk_rConvII[itau][itrack]);
            v_trk_nInnermostPixelHits.push_back(reader_trk_nInnermostPixelHits[itau][itrack]);
            v_trk_nPixelSharedHits.push_back(reader_trk_nPixelSharedHits[itau][itrack]);
            v_trk_nSCTSharedHits.push_back(reader_trk_nSCTSharedHits[itau][itrack]);
            v_trk_nPixelHits.push_back(reader_trk_nPixelHits[itau][itrack]);
            v_trk_nSiHits.push_back(reader_trk_nSiHits[itau][itrack]);
            v_trk_eProbabilityHT.push_back(reader_trk_eProbabilityHT[itau][itrack]);

            auto trackClassification = reader_trk_trackClassification[itau][itrack];
            v_trk_trackClassification.push_back(trackClassification);
            v_trk_isCharged.push_back(trackClassification == 3 ? 1 : 0);
            v_trk_isConversion.push_back(trackClassification == 2 ? 1 : 0);
            v_trk_isFakeOrIsolation.push_back(trackClassification == 1 ? 1 : 0);

            v_trk_nPixelHoles.push_back(reader_trk_nPixelHoles[itau][itrack]);
            v_trk_nPixelDead.push_back(reader_trk_nPixelDead[itau][itrack]);
            v_trk_nSCTHits.push_back(reader_trk_nSCTHits[itau][itrack]);
            v_trk_nSCTHoles.push_back(reader_trk_nSCTHoles[itau][itrack]);
            v_trk_nSCTDead.push_back(reader_trk_nSCTDead[itau][itrack]);
            v_trk_nSCTDoubleHoles.push_back(reader_trk_nSCTDoubleHoles[itau][itrack]);
            v_trk_nTRTHits.push_back(reader_trk_nTRTHits[itau][itrack]);
            v_trk_nTRTDead.push_back(reader_trk_nTRTDead[itau][itrack]);
        }
        
        for (size_t icluster = 0; icluster < reader_cls_e[itau].size(); icluster++)
        {
            v_cls_e.push_back(reader_cls_e[itau][icluster]);
            v_cls_et.push_back(reader_cls_et[itau][icluster]);
            v_cls_eta.push_back(reader_cls_eta[itau][icluster]);
            v_cls_phi.push_back(reader_cls_phi[itau][icluster]);
            v_cls_psfrac.push_back(reader_cls_psfrac[itau][icluster]);
            v_cls_em1frac.push_back(reader_cls_em1frac[itau][icluster]);
            v_cls_em2frac.push_back(reader_cls_em2frac[itau][icluster]);
            v_cls_em3frac.push_back(reader_cls_em3frac[itau][icluster]);
            v_cls_dRJetSeedAxis.push_back(reader_cls_dRJetSeedAxis[itau][icluster]);
            v_cls_EM_PROBABILITY.push_back(reader_cls_EM_PROBABILITY[itau][icluster]);
            v_cls_SECOND_R.push_back(reader_cls_SECOND_R[itau][icluster]);
            v_cls_SECOND_LAMBDA.push_back(reader_cls_SECOND_LAMBDA[itau][icluster]);
            v_cls_FIRST_ENG_DENS.push_back(reader_cls_FIRST_ENG_DENS[itau][icluster]);
            v_cls_CENTER_LAMBDA.push_back(reader_cls_CENTER_LAMBDA[itau][icluster]);
            v_cls_ENG_FRAC_MAX.push_back(reader_cls_ENG_FRAC_MAX[itau][icluster]);
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
