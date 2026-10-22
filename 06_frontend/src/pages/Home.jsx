import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import Button from '../components/common/Button';
import Reveal from '../components/common/Reveal';
import {
  IconActivity,
  IconArrowRight,
  IconDatabase,
  IconNetwork,
  IconPlayCircle,
  IconRefreshCw,
  IconShield,
  IconSparkles,
} from '../components/common/Icons';

const FEATURES = [
  {
    to: '/fleet',
    icon: IconActivity,
    tone: '',
    title: 'Fleet Intelligence',
    badge: 'C1',
    description:
      'Detect subtle behavioural drift across the HDD fleet early, with GenAI-assisted monitoring and anomaly reporting.',
  },
  {
    to: '/digital-twin',
    icon: IconRefreshCw,
    tone: 'indigo',
    title: 'Digital Twin',
    badge: 'C2',
    description:
      'Simulate drive ageing and run what-if scenarios in a calibrated digital twin before touching production hardware.',
  },
  {
    to: '/causal-xai',
    icon: IconNetwork,
    tone: 'green',
    title: 'Causal XAI',
    badge: 'C3',
    description:
      'Go beyond black-box scores — causal discovery and SHAP explanations show exactly why a drive is at risk.',
  },
  {
    to: '/safe-rl',
    icon: IconShield,
    tone: 'amber',
    title: 'Safe Optimization',
    badge: 'C4',
    description:
      'Risk-aware Safe-RL policies that migrate data only when the reliability gain outweighs the migration risk.',
  },
];

const STEPS = [
  {
    icon: IconDatabase,
    title: 'Monitor',
    description: 'Continuously ingest SMART telemetry and behavioural signals from every drive in the fleet.',
  },
  {
    icon: IconSparkles,
    title: 'Explain',
    description: 'Causal discovery and SHAP analysis surface root causes, not just raw failure probabilities.',
  },
  {
    icon: IconRefreshCw,
    title: 'Simulate',
    description: 'The digital twin replays and forecasts drive health under different intervention scenarios.',
  },
  {
    icon: IconShield,
    title: 'Act safely',
    description: 'Safety-filtered RL policies trigger migrations and healing actions within defined risk bounds.',
  },
];

const Home = () => {
  const { isAuthenticated } = useAuth();
  const startTo = isAuthenticated ? '/dashboard' : '/login';

  return (
    <>
      <section className="ui-hero">
        <div className="ui-container ui-hero__inner">
          <span className="ui-hero__badge">
            <IconSparkles size={14} />
            AI-Driven Reliability Engineering
          </span>
          <h1 className="ui-hero__title">
            Predict, explain and <span className="ui-grad-text">self-heal</span> HDD failures
          </h1>
          <p className="ui-hero__sub">
            A unified research framework that combines behavioural drift detection, digital twins, causal
            explainability and safe reinforcement learning to keep hard disk drives reliable — before they fail.
          </p>
          <div className="ui-hero__actions">
            <Link to={startTo}>
              <Button size="lg" rightIcon={<IconArrowRight size={18} />}>
                Get Started
              </Button>
            </Link>
            <a href="#features">
              <Button size="lg" variant="secondary">
                Explore modules
              </Button>
            </a>
          </div>
          <div className="ui-hero__stats">
            <div className="ui-hero__stat">
              <div className="ui-hero__stat-value">4</div>
              <div className="ui-hero__stat-label">Research components</div>
            </div>
            <div className="ui-hero__stat">
              <div className="ui-hero__stat-value">24/7</div>
              <div className="ui-hero__stat-label">Twin simulation</div>
            </div>
            <div className="ui-hero__stat">
              <div className="ui-hero__stat-value">Root-cause</div>
              <div className="ui-hero__stat-label">Explainable alerts</div>
            </div>
            <div className="ui-hero__stat">
              <div className="ui-hero__stat-value">Safe-RL</div>
              <div className="ui-hero__stat-label">Guarded actions</div>
            </div>
          </div>
        </div>
      </section>

      <section className="ui-section ui-section--alt" id="features">
        <div className="ui-container">
          <Reveal className="ui-section__head">
            <span className="ui-eyebrow">Framework modules</span>
            <h2 className="ui-section__title">Four components, one pipeline</h2>
            <p className="ui-section__sub">
              Each research component solves one stage of the reliability lifecycle and plugs into a shared
              FastAPI backend and dashboard.
            </p>
          </Reveal>
          <div className="ui-feature-grid">
            {FEATURES.map((feature) => {
              const Icon = feature.icon;
              return (
                <Reveal key={feature.to} className="ui-card ui-card--hover ui-feature">
                  <div className={`ui-feature__icon ${feature.tone ? `ui-feature__icon--${feature.tone}` : ''}`.trim()}>
                    <Icon size={22} />
                  </div>
                  <h3 className="ui-feature__title">
                    {feature.title}
                    <span className="ui-feature__badge">{feature.badge}</span>
                  </h3>
                  <p className="ui-feature__desc">{feature.description}</p>
                  <Link to={startTo} className="ui-feature__link">
                    Explore <IconArrowRight size={15} />
                  </Link>
                </Reveal>
              );
            })}
          </div>
        </div>
      </section>

      <section className="ui-section">
        <div className="ui-container">
          <Reveal className="ui-section__head">
            <span className="ui-eyebrow">How it works</span>
            <h2 className="ui-section__title">From raw telemetry to safe action</h2>
            <p className="ui-section__sub">
              A closed-loop pipeline that observes, explains, simulates and optimises — with human-readable
              evidence at every step.
            </p>
          </Reveal>
          <div className="ui-steps">
            {STEPS.map((step, index) => {
              const Icon = step.icon;
              return (
                <Reveal key={step.title} className="ui-card ui-step">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div className="ui-step__num">{index + 1}</div>
                    <Icon size={20} style={{ color: 'var(--ui-text-muted)' }} />
                  </div>
                  <h3 className="ui-step__title">{step.title}</h3>
                  <p className="ui-step__desc">{step.description}</p>
                </Reveal>
              );
            })}
          </div>
        </div>
      </section>

      <section className="ui-section ui-section--alt">
        <div className="ui-container">
          <Reveal className="ui-cta-band">
            <IconPlayCircle size={34} style={{ color: 'var(--ui-primary)' }} />
            <h2 className="ui-cta-band__title">Ready to explore the framework?</h2>
            <p className="ui-cta-band__sub">
              Sign in with the demo account to open the dashboard and browse every research module, live
              experiments and causal explanations.
            </p>
            <Link to={startTo}>
              <Button size="lg" rightIcon={<IconArrowRight size={18} />}>
                {isAuthenticated ? 'Open dashboard' : 'Get Started'}
              </Button>
            </Link>
          </Reveal>
        </div>
      </section>
    </>
  );
};

export default Home;
