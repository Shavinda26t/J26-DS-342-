import React, { useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import authService from '../services/authService';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import { IconAlertCircle, IconInfo, IconLock } from '../components/common/Icons';

const validate = (form) => {
  const errors = {};
  if (!form.email.trim()) {
    errors.email = 'Email is required.';
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email.trim())) {
    errors.email = 'Enter a valid email address.';
  }
  if (!form.password) {
    errors.password = 'Password is required.';
  }
  return errors;
};

const Login = () => {
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || '/dashboard';

  const [form, setForm] = useState({ email: '', password: '', remember: true });
  const [errors, setErrors] = useState({});
  const [formError, setFormError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  if (isAuthenticated) {
    return <Navigate to={from} replace />;
  }

  const updateField = (event) => {
    const { name, value, type, checked } = event.target;
    setForm((current) => ({ ...current, [name]: type === 'checkbox' ? checked : value }));
    if (errors[name]) {
      setErrors((current) => ({ ...current, [name]: undefined }));
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setFormError(null);
    const nextErrors = validate(form);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) return;

    setSubmitting(true);
    try {
      await login({ email: form.email, password: form.password, remember: form.remember });
      navigate(from, { replace: true });
    } catch (error) {
      setFormError(error.message || 'Login failed. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="ui-auth-card">
      <div className="ui-auth-card__head">
        <h1 className="ui-auth-card__title">Welcome back</h1>
        <p className="ui-auth-card__sub">Sign in to access the research dashboard</p>
      </div>

      {authService.AUTH_MODE === 'demo' && (
        <div className="ui-demo-note" role="note">
          <IconInfo size={16} />
          <span>
            <strong>Demo authentication.</strong> No production backend login exists yet. Sign in with{' '}
            <span className="ui-mono">{authService.demoCredentials.email}</span> /{' '}
            <span className="ui-mono">{authService.demoCredentials.password}</span>. This is client-side demo
            protection only and must not be treated as real security.
          </span>
        </div>
      )}

      {formError && (
        <div className="ui-alert ui-alert--error" role="alert" style={{ marginBottom: 18 }}>
          <IconAlertCircle size={16} />
          {formError}
        </div>
      )}

      <form className="ui-form" onSubmit={handleSubmit} noValidate>
        <Input
          label="Email"
          type="email"
          name="email"
          placeholder="you@example.com"
          autoComplete="email"
          value={form.email}
          onChange={updateField}
          error={errors.email}
        />
        <Input
          label="Password"
          type="password"
          name="password"
          placeholder="Enter your password"
          autoComplete="current-password"
          passwordToggle
          value={form.password}
          onChange={updateField}
          error={errors.password}
        />

        <label className="ui-checkbox">
          <input type="checkbox" name="remember" checked={form.remember} onChange={updateField} />
          Remember me on this device
        </label>

        <Button type="submit" fullWidth loading={submitting} leftIcon={<IconLock size={16} />}>
          {submitting ? 'Signing in…' : 'Sign in'}
        </Button>
      </form>

      <div className="ui-auth-card__footer">
        <Link to="/">← Return to the home page</Link>
      </div>
    </div>
  );
};

export default Login;
