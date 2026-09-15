import Header from "./components/Header";
import Upload from "./screens/Upload";
import Confirm from "./screens/Confirm";
import Analyzing from "./screens/Analyzing";
import Results from "./screens/Results";
import { useAnalysis } from "./useAnalysis";

export default function App() {
  const a = useAnalysis();

  return (
    <div className="min-h-screen bg-blush">
      <Header />

      {a.screen === "upload" && (
        <Upload onFile={a.upload} busy={a.busy} error={a.error} />
      )}

      {a.screen === "confirm" && (
        <Confirm
          preview={a.preview}
          detection={a.detection}
          error={a.error}
          onRun={a.run}
          onBack={a.reset}
        />
      )}

      {a.screen === "analyzing" && <Analyzing stages={a.stages} />}

      {a.screen === "results" && (
        <Results preview={a.preview} result={a.result} onReset={a.reset} />
      )}
    </div>
  );
}
