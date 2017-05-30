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
    using vuint8 = vector<uint8_t>;
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
    TRA_float reader_eta = {fReader, "TauJetsAuxDyn.eta"};
    TRA_float reader_phi = {fReader, "TauJetsAuxDyn.phi"};

    // TauJets variables
    TRA_double reader_mu = {fReader, "TauJetsAuxDyn.mu"};
    TRA_int reader_nVtxPU = {fReader, "TauJetsAuxDyn.nVtxPU"};
    TRA_int reader_PanTau_DecayMode = {fReader, "TauJetsAuxDyn.PanTau_DecayMode"};
    TRA_float reader_jet_Px = {fReader, "TauJetsAuxDyn.jet_Px"};
    TRA_float reader_jet_Py = {fReader, "TauJetsAuxDyn.jet_Py"};
    TRA_float reader_jet_Pz = {fReader, "TauJetsAuxDyn.jet_Pz"};
    TRA_float reader_jet_E = {fReader, "TauJetsAuxDyn.jet_E"};
    TRA_float reader_jet_Phi = {fReader, "TauJetsAuxDyn.jet_Phi"};
    TRA_float reader_jet_Eta = {fReader, "TauJetsAuxDyn.jet_Eta"};
    TRA_uint8 reader_jet_nChargedPFOs = {fReader, "TauJetsAuxDyn.jet_nChargedPFOs"};
    TRA_uint8 reader_jet_nNeutralPFOs = {fReader, "TauJetsAuxDyn.jet_nNeutralPFOs"};

    // PFO variables
    TRA_vfloat reader_pfo_chargedPx = {fReader, "TauJetsAuxDyn.pfo_chargedPx"};
    TRA_vfloat reader_pfo_chargedPy = {fReader, "TauJetsAuxDyn.pfo_chargedPy"};
    TRA_vfloat reader_pfo_chargedPz = {fReader, "TauJetsAuxDyn.pfo_chargedPz"};
    TRA_vfloat reader_pfo_chargedE = {fReader, "TauJetsAuxDyn.pfo_chargedE"};
    TRA_vfloat reader_pfo_chargedPhi = {fReader, "TauJetsAuxDyn.pfo_chargedPhi"};
    TRA_vfloat reader_pfo_chargedEta = {fReader, "TauJetsAuxDyn.pfo_chargedEta"};

    TRA_vfloat reader_pfo_neutralPx = {fReader, "TauJetsAuxDyn.pfo_neutralPx"};
    TRA_vfloat reader_pfo_neutralPy = {fReader, "TauJetsAuxDyn.pfo_neutralPy"};
    TRA_vfloat reader_pfo_neutralPz = {fReader, "TauJetsAuxDyn.pfo_neutralPz"};
    TRA_vfloat reader_pfo_neutralE = {fReader, "TauJetsAuxDyn.pfo_neutralE"};
    TRA_vfloat reader_pfo_neutralPhi = {fReader, "TauJetsAuxDyn.pfo_neutralPhi"};
    TRA_vfloat reader_pfo_neutralEta = {fReader, "TauJetsAuxDyn.pfo_neutralEta"};

    TRA_vfloat reader_pfo_neutralPi0BDT = {fReader, "TauJetsAuxDyn.pfo_neutralPi0BDT"};
    TRA_vuint8 reader_pfo_neutralNHitsInEM1 = {fReader, "TauJetsAuxDyn.pfo_neutralNHitsInEM1"};


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
    int v_PanTau_DecayMode;
    float v_jet_Px;
    float v_jet_Py;
    float v_jet_Pz;
    float v_jet_E;
    float v_jet_Phi;
    float v_jet_Eta;
    uint8_t v_jet_nChargedPFOs;
    uint8_t v_jet_nNeutralPFOs;

    // PFO variables
    vfloat v_pfo_chargedPx;
    vfloat v_pfo_chargedPy;
    vfloat v_pfo_chargedPz;
    vfloat v_pfo_chargedE;
    vfloat v_pfo_chargedPhi;
    vfloat v_pfo_chargedEta;

    vfloat v_pfo_neutralPx;
    vfloat v_pfo_neutralPy;
    vfloat v_pfo_neutralPz;
    vfloat v_pfo_neutralE;
    vfloat v_pfo_neutralPhi;
    vfloat v_pfo_neutralEta;
    vfloat v_pfo_neutralPi0BDT;
    vuint8 v_pfo_neutralNHitsInEM1;

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
    TBranch *b_PanTau_DecayMode = 0;
    TBranch *b_jet_Px = 0;
    TBranch *b_jet_Py = 0;
    TBranch *b_jet_Pz = 0;
    TBranch *b_jet_E = 0;
    TBranch *b_jet_Phi = 0;
    TBranch *b_jet_Eta = 0;
    TBranch *b_jet_nChargedPFOs = 0;
    TBranch *b_jet_nNeutralPFOs = 0;

    // PFO branches
    TBranch *b_pfo_chargedPx = 0;
    TBranch *b_pfo_chargedPy = 0;
    TBranch *b_pfo_chargedPz = 0;
    TBranch *b_pfo_chargedE = 0;
    TBranch *b_pfo_chargedPhi = 0;
    TBranch *b_pfo_chargedEta = 0;

    TBranch *b_pfo_neutralPx = 0;
    TBranch *b_pfo_neutralPy = 0;
    TBranch *b_pfo_neutralPz = 0;
    TBranch *b_pfo_neutralE = 0;
    TBranch *b_pfo_neutralPhi = 0;
    TBranch *b_pfo_neutralEta = 0;
    TBranch *b_pfo_neutralPi0BDT = 0;
    TBranch *b_pfo_neutralNHitsInEM1 = 0;

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
