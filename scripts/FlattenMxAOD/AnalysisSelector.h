// TODO: Dynamically load variables
#ifndef AnalysisSelector_h
#define AnalysisSelector_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

// Headers needed by this particular selector
#include <vector>
#include <string>

class AnalysisSelector : public TSelector
{
  public:
    TTreeReader fReader; //!the tree reader
    TTree *fChain = 0;   //!pointer to the analyzed TTree or TChain
    TFile *fOutFile = 0;
    TTree *fOutTree = 0;
    std::string fout_name;
    bool deco_truth;
    
    // Readers to access the data (delete the ones you do not need).
    using vfloat = vector<float>;
    using TRA_int = TTreeReaderArray<int>;
    using TRA_uint8 = TTreeReaderArray<uint8_t>;
    using TRA_float = TTreeReaderArray<float>;
    using TRA_double = TTreeReaderArray<double>;
    using TRA_vfloat = TTreeReaderArray<vfloat>;
    using TRA_vuint8 = TTreeReaderArray<vector<uint8_t>>;
    
    // Truth variables
    TTreeReaderArray<unsigned long> *reader_truthProng = 0;
    TTreeReaderArray<double> *reader_truthEtaVis = 0;
    TTreeReaderArray<double> *reader_truthPtVis = 0;
    TTreeReaderArray<char> *reader_IsTruthMatched = 0;
    TTreeReaderArray<unsigned long> *reader_truthDecayMode = 0;

    // TauJets variables
    TRA_int reader_nTracks = {fReader, "TauJetsAuxDyn.nTracks"};
    TRA_float reader_pt = {fReader, "TauJetsAuxDyn.pt"};
    TRA_float reader_ptJetSeed = {fReader, "TauJetsAuxDyn.trk_ptJetSeed"};
    TRA_float reader_etaJetSeed = {fReader, "TauJetsAuxDyn.trk_etaJetSeed"};
    TRA_float reader_phiJetSeed = {fReader, "TauJetsAuxDyn.trk_phiJetSeed"};
    TRA_float reader_eta = {fReader, "TauJetsAuxDyn.eta"};
    TRA_float reader_phi = {fReader, "TauJetsAuxDyn.phi"};
    
    // TauJets variables
    TRA_double reader_mu = {fReader, "TauJetsAuxDyn.mu"};
    TRA_int reader_nVtxPU = {fReader, "TauJetsAuxDyn.nVtxPU"};
    TRA_float reader_centFrac = {fReader, "TauJetsAuxDyn.centFrac"};
    TRA_float reader_EMPOverTrkSysP = {fReader, "TauJetsAuxDyn.EMPOverTrkSysP"};
    TRA_float reader_innerTrkAvgDist = {fReader, "TauJetsAuxDyn.innerTrkAvgDist"};
    TRA_float reader_ptRatioEflowApprox = {fReader, "TauJetsAuxDyn.ptRatioEflowApprox"};
    TRA_float reader_dRmax = {fReader, "TauJetsAuxDyn.dRmax"};
    TRA_float reader_trFlightPathSig = {fReader, "TauJetsAuxDyn.trFlightPathSig"};
    TRA_float reader_mEflowApprox = {fReader, "TauJetsAuxDyn.mEflowApprox"};
    TRA_float reader_SumPtTrkFrac = {fReader, "TauJetsAuxDyn.SumPtTrkFrac"};
    TRA_float reader_ipSigLeadTrk = {fReader, "TauJetsAuxDyn.ipSigLeadTrk"};
    TRA_float reader_massTrkSys = {fReader, "TauJetsAuxDyn.massTrkSys"};
    TRA_float reader_etOverPtLeadTrk = {fReader, "TauJetsAuxDyn.etOverPtLeadTrk"};
    TRA_float reader_ptIntermediateAxis = {fReader, "TauJetsAuxDyn.ptIntermediateAxis"};
    
    // Track-based variables
    TRA_uint8 reader_trk_nTracksTotal = {fReader, "TauJetsAuxDyn.trk_nTracksTotal"};
    TRA_vfloat reader_trk_pt = {fReader, "TauJetsAuxDyn.trk_pt"};
    TRA_vfloat reader_trk_qOverP = {fReader, "TauJetsAuxDyn.trk_qOverP"};
    TRA_vfloat reader_trk_eta = {fReader, "TauJetsAuxDyn.trk_eta"};
    TRA_vfloat reader_trk_phi = {fReader, "TauJetsAuxDyn.trk_phi"};
    TRA_vfloat reader_trk_eta_at_emcal = {fReader, "TauJetsAuxDyn.trk_eta_at_emcal"};
    TRA_vfloat reader_trk_phi_at_emcal = {fReader, "TauJetsAuxDyn.trk_phi_at_emcal"};
    TRA_vfloat reader_trk_z0sinThetaTJVA = {fReader, "TauJetsAuxDyn.trk_z0sinThetaTJVA"};
    TRA_vfloat reader_trk_d0 = {fReader, "TauJetsAuxDyn.trk_d0"};
    TRA_vfloat reader_trk_dRJetSeedAxis = {fReader, "TauJetsAuxDyn.trk_dRJetSeedAxis"};
    TRA_vfloat reader_trk_rConvII = {fReader, "TauJetsAuxDyn.trk_rConvII"};
    TRA_vuint8 reader_trk_nInnermostPixelHits = {fReader, "TauJetsAuxDyn.trk_nInnermostPixelHits"};
    TRA_vuint8 reader_trk_nPixelSharedHits = {fReader, "TauJetsAuxDyn.trk_nPixelSharedHits"};
    TRA_vuint8 reader_trk_nSCTSharedHits = {fReader, "TauJetsAuxDyn.trk_nSCTSharedHits"};
    TRA_vuint8 reader_trk_nPixelHits = {fReader, "TauJetsAuxDyn.trk_nPixelHits"};
    TRA_vuint8 reader_trk_nSiHits = {fReader, "TauJetsAuxDyn.trk_nSiHits"};
    TRA_vfloat reader_trk_eProbabilityHT = {fReader, "TauJetsAuxDyn.trk_eProbabilityHT"};
    TRA_vuint8 reader_trk_trackClassification = {fReader, "TauJetsAuxDyn.trk_classification"};

    TRA_vuint8 reader_trk_nPixelHoles = {fReader, "TauJetsAuxDyn.trk_nPixelHoles"};
    TRA_vuint8 reader_trk_nPixelDead = {fReader, "TauJetsAuxDyn.trk_nPixelDead"};
    TRA_vuint8 reader_trk_nSCTHits = {fReader, "TauJetsAuxDyn.trk_nSCTHits"};
    TRA_vuint8 reader_trk_nSCTHoles = {fReader, "TauJetsAuxDyn.trk_nSCTHoles"};
    TRA_vuint8 reader_trk_nSCTDead = {fReader, "TauJetsAuxDyn.trk_nSCTDead"};
    TRA_vuint8 reader_trk_nSCTDoubleHoles = {fReader, "TauJetsAuxDyn.trk_nSCTDoubleHoles"};
    TRA_vuint8 reader_trk_nTRTHits = {fReader, "TauJetsAuxDyn.trk_nTRTHits"};
    TRA_vuint8 reader_trk_nTRTDead = {fReader, "TauJetsAuxDyn.trk_nTRTDead"};

    // Cluster variables
    TRA_uint8 reader_cls_nClustersTotal = {fReader, "TauJetsAuxDyn.cls_nClustersTotal"};
    TRA_vfloat reader_cls_e = {fReader, "TauJetsAuxDyn.cls_e"};
    TRA_vfloat reader_cls_et = {fReader, "TauJetsAuxDyn.cls_et"};
    TRA_vfloat reader_cls_eta = {fReader, "TauJetsAuxDyn.cls_eta"};
    TRA_vfloat reader_cls_phi = {fReader, "TauJetsAuxDyn.cls_phi"};
    TRA_vfloat reader_cls_psfrac = {fReader, "TauJetsAuxDyn.cls_psfrac"};
    TRA_vfloat reader_cls_em1frac = {fReader, "TauJetsAuxDyn.cls_em1frac"};
    TRA_vfloat reader_cls_em2frac = {fReader, "TauJetsAuxDyn.cls_em2frac"};
    TRA_vfloat reader_cls_em3frac = {fReader, "TauJetsAuxDyn.cls_em3frac"};
    TRA_vfloat reader_cls_dRJetSeedAxis = {fReader, "TauJetsAuxDyn.cls_dRJetSeedAxis"};
    TRA_vfloat reader_cls_EM_PROBABILITY = {fReader, "TauJetsAuxDyn.cls_EM_PROBABILITY"};
    TRA_vfloat reader_cls_SECOND_R = {fReader, "TauJetsAuxDyn.cls_SECOND_R"};
    TRA_vfloat reader_cls_SECOND_LAMBDA = {fReader, "TauJetsAuxDyn.cls_SECOND_LAMBDA"};
    TRA_vfloat reader_cls_FIRST_ENG_DENS = {fReader, "TauJetsAuxDyn.cls_FIRST_ENG_DENS"};
    TRA_vfloat reader_cls_CENTER_LAMBDA = {fReader, "TauJetsAuxDyn.cls_CENTER_LAMBDA"};
    TRA_vfloat reader_cls_ENG_FRAC_MAX = {fReader, "TauJetsAuxDyn.cls_ENG_FRAC_MAX"};


    // Truth variables
    unsigned long v_truthProng;
    double v_truthEtaVis;
    double v_truthPtVis;
    char v_IsTruthMatched;
    unsigned long v_truthDecayMode;

    // TauJets variables
    int v_nTracks;
    float v_pt;
    float v_ptJetSeed;
    float v_etaJetSeed;
    float v_phiJetSeed;
    float v_eta;
    float v_phi;
    double v_mu;
    int v_nVtxPU;
    float v_centFrac;
    float v_EMPOverTrkSysP;
    float v_innerTrkAvgDist;
    float v_ptRatioEflowApprox;
    float v_dRmax;
    float v_trFlightPathSig;
    float v_mEflowApprox;
    float v_SumPtTrkFrac;
    float v_absipSigLeadTrk;
    float v_massTrkSys;
    float v_etOverPtLeadTrk;
    float v_ptIntermediateAxis;

    // Track variables
    // Total number of tracks (charged, conversion, isolation, fake)
    int v_trk_nTracksTotal;
    std::vector<float> v_trk_pt;
    std::vector<float> v_trk_qOverP;
    std::vector<float> v_trk_eta;
    std::vector<float> v_trk_phi;
    std::vector<float> v_trk_eta_at_emcal;
    std::vector<float> v_trk_phi_at_emcal;
    std::vector<float> v_trk_z0sinThetaTJVA;
    std::vector<float> v_trk_d0;
    std::vector<float> v_trk_dRJetSeedAxis;
    std::vector<float> v_trk_rConvII;
    std::vector<uint8_t> v_trk_nInnermostPixelHits;
    std::vector<uint8_t> v_trk_nPixelSharedHits;
    std::vector<uint8_t> v_trk_nSCTSharedHits;
    std::vector<uint8_t> v_trk_nPixelHits;
    std::vector<uint8_t> v_trk_nSiHits;
    std::vector<float> v_trk_eProbabilityHT;
    std::vector<uint8_t> v_trk_trackClassification;
    std::vector<uint8_t> v_trk_isCharged;
    std::vector<uint8_t> v_trk_isConversion;
    std::vector<uint8_t> v_trk_isFakeOrIsolation;

    std::vector<uint8_t> v_trk_nPixelHoles;
    std::vector<uint8_t> v_trk_nPixelDead;
    std::vector<uint8_t> v_trk_nSCTHits;
    std::vector<uint8_t> v_trk_nSCTHoles;
    std::vector<uint8_t> v_trk_nSCTDead;
    std::vector<uint8_t> v_trk_nSCTDoubleHoles;
    std::vector<uint8_t> v_trk_nTRTHits;
    std::vector<uint8_t> v_trk_nTRTDead;

    // Cluster variables
    int v_cls_nClustersTotal;
    vfloat v_cls_e;
    vfloat v_cls_et;
    vfloat v_cls_eta;
    vfloat v_cls_phi;
    vfloat v_cls_psfrac;
    vfloat v_cls_em1frac;
    vfloat v_cls_em2frac;
    vfloat v_cls_em3frac;
    vfloat v_cls_dRJetSeedAxis;
    vfloat v_cls_EM_PROBABILITY;
    vfloat v_cls_SECOND_R;
    vfloat v_cls_SECOND_LAMBDA;
    vfloat v_cls_FIRST_ENG_DENS;
    vfloat v_cls_CENTER_LAMBDA;
    vfloat v_cls_ENG_FRAC_MAX;


    // output branches
    // Truth branches
    TBranch *b_truthProng = 0;
    TBranch *b_truthEtaVis = 0;
    TBranch *b_truthPtVis = 0;
    TBranch *b_IsTruthMatched = 0;
    TBranch *b_truthDecayMode = 0;
    
    // TauJets branches
    TBranch *b_pt = 0;
    TBranch *b_eta = 0;
    TBranch *b_phi = 0;
    TBranch *b_nTracks = 0;
    TBranch *b_ptJetSeed = 0;
    TBranch *b_etaJetSeed = 0;
    TBranch *b_phiJetSeed = 0;
    TBranch *b_mu = 0;
    TBranch *b_nVtxPU = 0;
    TBranch *b_centFrac = 0;
    TBranch *b_EMPOverTrkSysP = 0;
    TBranch *b_innerTrkAvgDist = 0;
    TBranch *b_ptRatioEflowApprox = 0;
    TBranch *b_dRmax = 0;
    TBranch *b_trFlightPathSig = 0;
    TBranch *b_mEflowApprox = 0;
    TBranch *b_SumPtTrkFrac = 0;
    TBranch *b_absipSigLeadTrk = 0;
    TBranch *b_massTrkSys = 0;
    TBranch *b_etOverPtLeadTrk = 0;
    TBranch *b_ptIntermediateAxis = 0;

    // Track branches
    // nTracksTotal
    TBranch *b_nTracksTotal = 0;
    TBranch *b_trk_pt = 0;
    TBranch *b_trk_qOverP = 0;
    TBranch *b_trk_eta = 0;
    TBranch *b_trk_phi = 0;
    TBranch *b_trk_eta_at_emcal = 0;
    TBranch *b_trk_phi_at_emcal = 0;
    TBranch *b_trk_z0sinThetaTJVA = 0;
    TBranch *b_trk_d0 = 0;
    TBranch *b_trk_dRJetSeedAxis = 0;
    TBranch *b_trk_rConvII = 0;
    TBranch *b_trk_nInnermostPixelHits = 0;
    TBranch *b_trk_nPixelSharedHits = 0;
    TBranch *b_trk_nSCTSharedHits = 0;
    TBranch *b_trk_nPixelHits = 0;
    TBranch *b_trk_nSiHits = 0;
    TBranch *b_trk_eProbabilityHT = 0;
    TBranch *b_trk_trackClassification = 0;
    TBranch *b_trk_isCharged = 0;
    TBranch *b_trk_isConversion = 0;
    TBranch *b_trk_isFakeOrIsolation = 0;

    TBranch *b_trk_nPixelHoles = 0;
    TBranch *b_trk_nPixelDead = 0;
    TBranch *b_trk_nSCTHits = 0;
    TBranch *b_trk_nSCTHoles = 0;
    TBranch *b_trk_nSCTDead = 0;
    TBranch *b_trk_nSCTDoubleHoles = 0;
    TBranch *b_trk_nTRTHits = 0;
    TBranch *b_trk_nTRTDead = 0;

    // Cluster branches
    TBranch *b_cls_nClustersTotal = 0;
    TBranch *b_cls_e = 0;
    TBranch *b_cls_et = 0;
    TBranch *b_cls_eta = 0;
    TBranch *b_cls_phi = 0;
    TBranch *b_cls_psfrac = 0;
    TBranch *b_cls_em1frac = 0;
    TBranch *b_cls_em2frac = 0;
    TBranch *b_cls_em3frac = 0;
    TBranch *b_cls_dRJetSeedAxis = 0;
    TBranch *b_cls_EM_PROBABILITY = 0;
    TBranch *b_cls_SECOND_R = 0;
    TBranch *b_cls_SECOND_LAMBDA = 0;
    TBranch *b_cls_FIRST_ENG_DENS = 0;
    TBranch *b_cls_CENTER_LAMBDA = 0;
    TBranch *b_cls_ENG_FRAC_MAX = 0;
    
    
    AnalysisSelector(TTree * /*tree*/ = 0){ fout_name = "temp.root"; }
    AnalysisSelector(std::string fout, bool truth) {
        fout_name = fout;
        deco_truth = truth;

        if (deco_truth) {
            reader_truthProng = new TTreeReaderArray<unsigned long>(fReader,
                "TauJetsAuxDyn.truthProng");
            reader_truthEtaVis = new TTreeReaderArray<double>(fReader,
                "TauJetsAuxDyn.truthEtaVis");
            reader_truthPtVis = new TTreeReaderArray<double>(fReader,
                "TauJetsAuxDyn.truthPtVis");
            reader_IsTruthMatched = new TTreeReaderArray<char>(fReader,
                "TauJetsAuxDyn.IsTruthMatched");
            reader_truthDecayMode = new TTreeReaderArray<unsigned long>(fReader,
                "TauJetsAuxDyn.truthDecayMode");
        }
    }
    virtual ~AnalysisSelector() {}
    virtual Int_t Version() const { return 2; }
    virtual void Begin(TTree *tree);
    virtual void SlaveBegin(TTree *tree);
    virtual void Init(TTree *tree);
    virtual Bool_t Notify();
    virtual Bool_t Process(Long64_t entry);
    virtual Int_t GetEntry(Long64_t entry, Int_t getall = 0) { return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0; }
    virtual void SetOption(const char *option) { fOption = option; }
    virtual void SetObject(TObject *obj) { fObject = obj; }
    virtual void SetInputList(TList *input) { fInput = input; }
    virtual TList *GetOutputList() const { return fOutput; }
    virtual void SlaveTerminate();
    virtual void Terminate();

    ClassDef(AnalysisSelector, 1);
};

#endif

#ifdef AnalysisSelector_cxx
void AnalysisSelector::Init(TTree *tree)
{
    // The Init() function is called when the selector needs to initialize
    // a new tree or chain. Typically here the reader is initialized.
    // It is normally not necessary to make changes to the generated
    // code, but the routine can be extended by the user if needed.
    // Init() will be called many times when running on PROOF
    // (once per file to be processed).

    fReader.SetTree(tree);
}

Bool_t AnalysisSelector::Notify()
{
    // The Notify() function is called when a new file is opened. This
    // can be either for a new TTree in a TChain or when when a new TTree
    // is started when using PROOF. It is normally not necessary to make changes
    // to the generated code, but the routine can be extended by the
    // user if needed. The return value is currently not used.

    return kTRUE;
}

#endif // #ifdef AnalysisSelector_cxx
