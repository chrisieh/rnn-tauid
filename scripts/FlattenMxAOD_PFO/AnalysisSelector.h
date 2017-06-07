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
    using vint = vector<int>;
    using vuint8 = vector<uint8_t>;
    using TRA_int = TTreeReaderArray<int>;
    using TRA_uint8 = TTreeReaderArray<uint8_t>;
    using TRA_float = TTreeReaderArray<float>;
    using TRA_double = TTreeReaderArray<double>;
    using TRA_vfloat = TTreeReaderArray<vfloat>;
    using TRA_vint = TTreeReaderArray<vint>;
    using TRA_vuint8 = TTreeReaderArray<vuint8>;

    // Truth variables
    TTreeReaderArray<unsigned long> *reader_truthProng = 0;
    TTreeReaderArray<double> *reader_truthEtaVis = 0;
    TTreeReaderArray<double> *reader_truthPtVis = 0;
    TTreeReaderArray<char> *reader_IsTruthMatched = 0;
    TTreeReaderArray<unsigned long> *reader_truthDecayMode = 0;

    // TauJets variables
    TRA_int reader_nTracks = {fReader, "TauJetsAuxDyn.nTracks"};
    TRA_float reader_pt = {fReader, "TauJetsAuxDyn.pt"};
    TRA_float reader_eta = {fReader, "TauJetsAuxDyn.eta"};
    TRA_float reader_phi = {fReader, "TauJetsAuxDyn.phi"};

    // TauJets variables
    TRA_double reader_mu = {fReader, "TauJetsAuxDyn.mu"};
    TRA_int reader_nVtxPU = {fReader, "TauJetsAuxDyn.nVtxPU"};
    TRA_int reader_PanTau_DecayModeProto = {fReader, "TauJetsAuxDyn.PanTau_DecayModeProto"};
    TRA_int reader_PanTau_DecayMode = {fReader, "TauJetsAuxDyn.PanTau_DecayMode"};
    TRA_float reader_jet_Pt = {fReader, "TauJetsAuxDyn.jet_Pt"};
    TRA_float reader_jet_Phi = {fReader, "TauJetsAuxDyn.jet_Phi"};
    TRA_float reader_jet_Eta = {fReader, "TauJetsAuxDyn.jet_Eta"};
    TRA_uint8 reader_jet_nChargedPFOs = {fReader, "TauJetsAuxDyn.jet_nChargedPFOs"};
    TRA_uint8 reader_jet_nNeutralPFOs = {fReader, "TauJetsAuxDyn.jet_nNeutralPFOs"};
    TRA_uint8 reader_jet_nShotPFOs = {fReader, "TauJetsAuxDyn.jet_nShotPFOs"};
    TRA_uint8 reader_jet_nHadronicPFOs = {fReader, "TauJetsAuxDyn.jet_nHadronicPFOs"};
    TRA_uint8 reader_jet_nConversion = {fReader, "TauJetsAuxDyn.jet_nConversion"};

    TRA_float reader_BDTJetScore = {fReader, "TauJetsAuxDyn.BDTJetScore"};
    TRA_float reader_BDTJetScoreSigTrans = {fReader, "TauJetsAuxDyn.BDTJetScoreSigTrans"};

    // PFO variables
    TRA_vfloat reader_pfo_chargedPt = {fReader, "TauJetsAuxDyn.pfo_chargedPt"};
    TRA_vfloat reader_pfo_chargedPhi = {fReader, "TauJetsAuxDyn.pfo_chargedPhi"};
    TRA_vfloat reader_pfo_chargedEta = {fReader, "TauJetsAuxDyn.pfo_chargedEta"};

    TRA_vfloat reader_pfo_neutralPt = {fReader, "TauJetsAuxDyn.pfo_neutralPt"};
    TRA_vfloat reader_pfo_neutralPhi = {fReader, "TauJetsAuxDyn.pfo_neutralPhi"};
    TRA_vfloat reader_pfo_neutralEta = {fReader, "TauJetsAuxDyn.pfo_neutralEta"};
    TRA_vfloat reader_pfo_neutralPi0BDT = {fReader, "TauJetsAuxDyn.pfo_neutralPi0BDT"};
    TRA_vuint8 reader_pfo_neutralNHitsInEM1 = {fReader, "TauJetsAuxDyn.pfo_neutralNHitsInEM1"};

    TRA_vfloat reader_pfo_neutralPtSub =
        {fReader, "TauJetsAuxDyn.pfo_neutralPtSub"};
    TRA_vfloat reader_pfo_neutral_SECOND_R =
        {fReader, "TauJetsAuxDyn.pfo_neutral_SECOND_R"};
    TRA_vfloat reader_pfo_neutral_SECOND_LAMBDA =
        {fReader, "TauJetsAuxDyn.pfo_neutral_SECOND_LAMBDA"};
    TRA_vfloat reader_pfo_neutral_CENTER_LAMBDA =
        {fReader, "TauJetsAuxDyn.pfo_neutral_CENTER_LAMBDA"};
    TRA_vfloat reader_pfo_neutral_ENG_FRAC_MAX =
        {fReader, "TauJetsAuxDyn.pfo_neutral_ENG_FRAC_MAX"};
    TRA_vfloat reader_pfo_neutral_ENG_FRAC_CORE =
        {fReader, "TauJetsAuxDyn.pfo_neutral_ENG_FRAC_CORE"};
    TRA_vfloat reader_pfo_neutral_SECOND_ENG_DENS =
        {fReader, "TauJetsAuxDyn.pfo_neutral_SECOND_ENG_DENS"};
    TRA_vint reader_pfo_neutral_NPosECells_EM1 =
        {fReader, "TauJetsAuxDyn.pfo_neutral_NPosECells_EM1"};
    TRA_vint reader_pfo_neutral_NPosECells_EM2 =
        {fReader, "TauJetsAuxDyn.pfo_neutral_NPosECells_EM2"};
    TRA_vfloat reader_pfo_neutral_secondEtaWRTClusterPosition_EM1 =
        {fReader, "TauJetsAuxDyn.pfo_neutral_secondEtaWRTClusterPosition_EM1"};
    TRA_vfloat reader_pfo_neutral_secondEtaWRTClusterPosition_EM2 =
        {fReader, "TauJetsAuxDyn.pfo_neutral_secondEtaWRTClusterPosition_EM2"};
    TRA_vfloat reader_pfo_neutral_energyfrac_EM1 =
        {fReader, "TauJetsAuxDyn.pfo_neutral_energyfrac_EM1"};
    TRA_vfloat reader_pfo_neutral_energyfrac_EM2 =
        {fReader, "TauJetsAuxDyn.pfo_neutral_energyfrac_EM2"};

    TRA_vfloat reader_pfo_neutralPt_BDTSort =
        {fReader, "TauJetsAuxDyn.pfo_neutralPt_BDTSort"};
    TRA_vfloat reader_pfo_neutralPhi_BDTSort =
        {fReader, "TauJetsAuxDyn.pfo_neutralPhi_BDTSort"};
    TRA_vfloat reader_pfo_neutralEta_BDTSort =
        {fReader, "TauJetsAuxDyn.pfo_neutralEta_BDTSort"};
    TRA_vfloat reader_pfo_neutralPi0BDT_BDTSort =
        {fReader, "TauJetsAuxDyn.pfo_neutralPi0BDT_BDTSort"};
    TRA_vuint8 reader_pfo_neutralNHitsInEM1_BDTSort =
        {fReader, "TauJetsAuxDyn.pfo_neutralNHitsInEM1_BDTSort"};

    TRA_vfloat reader_pfo_shotPt = {fReader, "TauJetsAuxDyn.pfo_shotPt"};
    TRA_vfloat reader_pfo_shotPhi = {fReader, "TauJetsAuxDyn.pfo_shotPhi"};
    TRA_vfloat reader_pfo_shotEta = {fReader, "TauJetsAuxDyn.pfo_shotEta"};

    TRA_vfloat reader_pfo_hadronicPt = {fReader, "TauJetsAuxDyn.pfo_hadronicPt"};
    TRA_vfloat reader_pfo_hadronicPhi = {fReader, "TauJetsAuxDyn.pfo_hadronicPhi"};
    TRA_vfloat reader_pfo_hadronicEta = {fReader, "TauJetsAuxDyn.pfo_hadronicEta"};

    TRA_vfloat reader_conv_pt = {fReader, "TauJetsAuxDyn.conv_pt"};
    TRA_vfloat reader_conv_phi = {fReader, "TauJetsAuxDyn.conv_phi"};
    TRA_vfloat reader_conv_eta = {fReader, "TauJetsAuxDyn.conv_eta"};
    TRA_vfloat reader_conv_phi_extrap = {fReader, "TauJetsAuxDyn.conv_phi_extrap"};
    TRA_vfloat reader_conv_eta_extrap = {fReader, "TauJetsAuxDyn.conv_eta_extrap"};

    // Truth variables
    unsigned long v_truthProng;
    double v_truthEtaVis;
    double v_truthPtVis;
    char v_IsTruthMatched;
    unsigned long v_truthDecayMode;

    // TauJets variables
    int v_nTracks;
    float v_pt;
    float v_eta;
    float v_phi;
    double v_mu;
    int v_nVtxPU;
    int v_PanTau_DecayModeProto;
    int v_PanTau_DecayMode;
    float v_jet_Pt;
    float v_jet_Phi;
    float v_jet_Eta;
    uint8_t v_jet_nChargedPFOs;
    uint8_t v_jet_nNeutralPFOs;
    uint8_t v_jet_nShotPFOs;
    uint8_t v_jet_nHadronicPFOs;
    uint8_t v_jet_nConversion;
    float v_BDTJetScore;
    float v_BDTJetScoreSigTrans;

    // PFO variables
    vfloat v_pfo_chargedPt;
    vfloat v_pfo_chargedPhi;
    vfloat v_pfo_chargedEta;

    vfloat v_pfo_neutralPt;
    vfloat v_pfo_neutralPhi;
    vfloat v_pfo_neutralEta;
    vfloat v_pfo_neutralPi0BDT;
    vuint8 v_pfo_neutralNHitsInEM1;

    vfloat v_pfo_neutralPtSub;
    vfloat v_pfo_neutral_SECOND_R;
    vfloat v_pfo_neutral_SECOND_LAMBDA;
    vfloat v_pfo_neutral_CENTER_LAMBDA;
    vfloat v_pfo_neutral_ENG_FRAC_MAX;
    vfloat v_pfo_neutral_ENG_FRAC_CORE;
    vfloat v_pfo_neutral_SECOND_ENG_DENS;
    vint v_pfo_neutral_NPosECells_EM1;
    vint v_pfo_neutral_NPosECells_EM2;
    vfloat v_pfo_neutral_secondEtaWRTClusterPosition_EM1;
    vfloat v_pfo_neutral_secondEtaWRTClusterPosition_EM2;
    vfloat v_pfo_neutral_energyfrac_EM1;
    vfloat v_pfo_neutral_energyfrac_EM2;

    vfloat v_pfo_neutralPt_BDTSort;
    vfloat v_pfo_neutralPhi_BDTSort;
    vfloat v_pfo_neutralEta_BDTSort;
    vfloat v_pfo_neutralPi0BDT_BDTSort;
    vuint8 v_pfo_neutralNHitsInEM1_BDTSort;

    vfloat v_pfo_shotPt;
    vfloat v_pfo_shotPhi;
    vfloat v_pfo_shotEta;

    vfloat v_pfo_hadronicPt;
    vfloat v_pfo_hadronicPhi;
    vfloat v_pfo_hadronicEta;

    vfloat v_conv_pt;
    vfloat v_conv_phi;
    vfloat v_conv_eta;
    vfloat v_conv_phi_extrap;
    vfloat v_conv_eta_extrap;

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
    TBranch *b_mu = 0;
    TBranch *b_nVtxPU = 0;
    TBranch *b_PanTau_DecayModeProto = 0;
    TBranch *b_PanTau_DecayMode = 0;
    TBranch *b_jet_Pt = 0;
    TBranch *b_jet_Phi = 0;
    TBranch *b_jet_Eta = 0;
    TBranch *b_jet_nChargedPFOs = 0;
    TBranch *b_jet_nNeutralPFOs = 0;
    TBranch *b_jet_nShotPFOs = 0;
    TBranch *b_jet_nHadronicPFOs = 0;
    TBranch *b_jet_nConversion = 0;
    TBranch *b_BDTJetScore = 0;
    TBranch *b_BDTJetScoreSigTrans = 0;

    // PFO branches
    TBranch *b_pfo_chargedPt = 0;
    TBranch *b_pfo_chargedPhi = 0;
    TBranch *b_pfo_chargedEta = 0;

    TBranch *b_pfo_neutralPt = 0;
    TBranch *b_pfo_neutralPhi = 0;
    TBranch *b_pfo_neutralEta = 0;
    TBranch *b_pfo_neutralPi0BDT = 0;
    TBranch *b_pfo_neutralNHitsInEM1 = 0;

    TBranch *b_pfo_neutralPtSub = 0;
    TBranch *b_pfo_neutral_SECOND_R = 0;
    TBranch *b_pfo_neutral_SECOND_LAMBDA = 0;
    TBranch *b_pfo_neutral_CENTER_LAMBDA = 0;
    TBranch *b_pfo_neutral_ENG_FRAC_MAX = 0;
    TBranch *b_pfo_neutral_ENG_FRAC_CORE = 0;
    TBranch *b_pfo_neutral_SECOND_ENG_DENS = 0;
    TBranch *b_pfo_neutral_NPosECells_EM1 = 0;
    TBranch *b_pfo_neutral_NPosECells_EM2 = 0;
    TBranch *b_pfo_neutral_secondEtaWRTClusterPosition_EM1 = 0;
    TBranch *b_pfo_neutral_secondEtaWRTClusterPosition_EM2 = 0;
    TBranch *b_pfo_neutral_energyfrac_EM1 = 0;
    TBranch *b_pfo_neutral_energyfrac_EM2 = 0;

    TBranch *b_pfo_neutralPt_BDTSort = 0;
    TBranch *b_pfo_neutralPhi_BDTSort = 0;
    TBranch *b_pfo_neutralEta_BDTSort = 0;
    TBranch *b_pfo_neutralPi0BDT_BDTSort = 0;
    TBranch *b_pfo_neutralNHitsInEM1_BDTSort = 0;

    TBranch *b_pfo_shotPt = 0;
    TBranch *b_pfo_shotPhi = 0;
    TBranch *b_pfo_shotEta = 0;

    TBranch *b_pfo_hadronicPt = 0;
    TBranch *b_pfo_hadronicPhi = 0;
    TBranch *b_pfo_hadronicEta = 0;

    TBranch *b_conv_pt = 0;
    TBranch *b_conv_phi = 0;
    TBranch *b_conv_eta = 0;
    TBranch *b_conv_phi_extrap = 0;
    TBranch *b_conv_eta_extrap = 0;

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
