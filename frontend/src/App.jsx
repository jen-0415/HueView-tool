import Header from "./components/Header";
import Upload from "./screens/Upload";
import Confirm from "./screens/Confirm";
import Analyzing from "./screens/Analyzing";
import Results from "./screens/Results";
import Evaluation from "./screens/Evaluation";
import { useAnalysis } from "./useAnalysis";
import { useView } from "./view";

export default function App() {
  const a = useAnalysis();
  // The analysis flow keeps its state while the Evaluation tab is open.
  const { tab, setTab } = useView();

  return (
    <div className="min-h-screen bg-blush">
      <Header tab={tab} onTab={setTab} />

      {tab === "evaluation" && <Evaluation />}

      {tab === "analyze" && a.screen === "upload" && (
        <Upload onFile={a.upload} busy={a.busy} error={a.error} />
      )}

      {tab === "analyze" && a.screen === "confirm" && (
        <Confirm
          preview={a.preview}
          detection={a.detection}
          error={a.error}
          onRun={a.run}
          onBack={a.reset}
        />
      )}

      {tab === "analyze" && a.screen === "analyzing" && <Analyzing />}

      {tab === "analyze" && a.screen === "results" && (
        <Results
          preview={a.preview}
          result={a.result}
          ssrPreview={a.ssrPreview}
          onReset={a.reset}
        />
      )}
    </div>
  );
}
