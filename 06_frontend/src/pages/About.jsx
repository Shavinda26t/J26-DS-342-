import React from 'react';
import { Link } from 'react-router-dom';
import Button from '../components/common/Button';
import Reveal from '../components/common/Reveal';
import {
  IconActivity,
  IconArrowRight,
  IconCheck,
  IconNetwork,
  IconRefreshCw,
  IconShield,
} from '../components/common/Icons';

const OBJECTIVES = [
  'Detect HDD failure risk early using behavioural drift analysis instead of threshold-only SMART rules.',
  'Explain every prediction with causal evidence so operators can trust and audit automated decisions.',
  'Simulate interventions in a digital twin before applying them to physical drives.',
  'Optimise migration and healing actions with safety-guarded reinforcement learning.',
];

const MODULES = [
  {
    icon: IconActivity,
    tone: '',
    title: 'Component 1 — Behavioural Drift & GenAI',
    description: 'Monitors population-level behaviour shifts and produces natural-language drift summaries.',
  },
  {
    icon: IconRefreshCw,
    tone: 'indigo',
    title: 'Component 2 — Digital Twin',
    description: 'Calibrated per-drive simulators for ageing forecasts and what-if scenario testing.',
  },
  {
    icon: IconNetwork,
    tone: 'green',
    title: 'Component 3 — Causal XAI',
    description: 'PCMCI causal discovery plus SHAP for root-cause panels and evidence-ranked factors.',
  },
  {
    icon: IconShield,
    tone: 'amber',
    title: 'Component 4 — Safe Deep RL',
    description: 'Risk-sensitive migration policies with hard safety constraints and holdout validation.',
  },
];

const About = () => (
  <>
    <section className="ui-page-hero">
      <div className="ui-container">
        <span className="ui-eyebrow">About the project</span>
        <h1 className="ui-page-hero__title">
          An end-to-end <span className="ui-grad-text">self-healing</span> storage framework
        </h1>
        <p className="ui-page-hero__sub">
          This final-year research project investigates how causal machine learning, digital twins and safe
          reinforcement learning can work together to predict and mitigate hard disk drive failures in data-centre
          fleets.
        </p>
      </div>
    </section>

    <section className="ui-section">
      <div className="ui-container ui-two-col">
        <Reveal className="ui-card">
          <div className="ui-prose">
            <h3>Research objectives</h3>
            <ul className="ui-list">
              {OBJECTIVES.map((objective) => (
                <li key={objective}>
                  <IconCheck size={16} />
                  <span>{objective}</span>
                </li>
              ))}
            </ul>
          </div>
        </Reveal>
        <Reveal className="ui-card">
          <div className="ui-prose">
            <h3>Architecture at a glance</h3>
            <p>
              Four research components are developed independently and integrated through a shared FastAPI backend
              (<span className="ui-mono">/api/v1</span>) with a React dashboard. JSON contracts in{' '}
              <span className="ui-mono">07_shared_contracts</span> keep every component decoupled, so modules can
              evolve without breaking the pipeline.
            </p>
            <p>
              The dashboard itself uses a centralized common UI architecture — shared header, footer, sidebar and
              layouts — so new research modules can be added without rebuilding the shell.
            </p>
            <Link to="/contact">
              <Button variant="secondary" rightIcon={<IconArrowRight size={16} />}>
                Contact the team
              </Button>
            </Link>
          </div>
        </Reveal>
      </div>
    </section>

    <section className="ui-section ui-section--alt">
      <div className="ui-container">
        <Reveal className="ui-section__head">
          <span className="ui-eyebrow">Components</span>
          <h2 className="ui-section__title">Research modules</h2>
          <p className="ui-section__sub">Each component is a self-contained research contribution with its own code, tests and results.</p>
        </Reveal>
        <div className="ui-feature-grid">
          {MODULES.map((module) => {
            const Icon = module.icon;
            return (
              <Reveal key={module.title} className="ui-card ui-feature">
                <div className={`ui-feature__icon ${module.tone ? `ui-feature__icon--${module.tone}` : ''}`.trim()}>
                  <Icon size={22} />
                </div>
                <h3 className="ui-feature__title">{module.title}</h3>
                <p className="ui-feature__desc">{module.description}</p>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  </>
);

export default About;
