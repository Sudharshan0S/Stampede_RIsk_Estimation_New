import { useEffect } from "react";
import { AppProvider, useApp } from "./context/AppContext";
import HeroHeader from "./components/layout/HeroHeader";
import NavBar from "./components/layout/NavBar";
import InputPage from "./pages/InputPage";
import AnalysisPage from "./pages/AnalysisPage";
import ResultsPage from "./pages/ResultsPage";
import AboutPage from "./pages/AboutPage";

function Shell() {
  const { tab, pingBackend } = useApp();

  useEffect(() => {
    pingBackend();
    const id = setInterval(pingBackend, 15000);
    return () => clearInterval(id);
  }, [pingBackend]);

  return (
    <div className="app-shell flex flex-col">
      <NavBar />
      <HeroHeader />
      <main className="flex-1 min-w-0 px-4 sm:px-6 lg:px-10 pb-12 sm:pb-16">
        <div className="mx-auto max-w-[1400px]">
          {tab === "input" && <InputPage />}
          {tab === "analysis" && <AnalysisPage />}
          {tab === "results" && <ResultsPage />}
          {tab === "about" && <AboutPage />}
        </div>
      </main>
      <footer className="safe-bottom px-4 sm:px-6 lg:px-10 pt-6 border-t border-white/[0.06] text-center text-xs text-slate-600">
        CrowdVision AI · Smart Crowd Intelligence · Runs entirely on your local backend
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <Shell />
    </AppProvider>
  );
}
