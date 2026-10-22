import React, { useId, useState } from 'react';
import { IconAlertCircle, IconEye, IconEyeOff } from './Icons';

const Input = ({
  label,
  error,
  hint,
  type = 'text',
  passwordToggle = false,
  className = '',
  id,
  ...rest
}) => {
  const autoId = useId();
  const inputId = id || autoId;
  const [visible, setVisible] = useState(false);
  const isPassword = type === 'password';
  const hasToggle = isPassword && passwordToggle;
  const resolvedType = hasToggle ? (visible ? 'text' : 'password') : type;

  const describedBy = error ? `${inputId}-error` : hint ? `${inputId}-hint` : undefined;

  return (
    <div className={`ui-field ${className}`}>
      {label && (
        <label className="ui-label" htmlFor={inputId}>
          {label}
        </label>
      )}
      <div className="ui-input-wrap">
        <input
          id={inputId}
          type={resolvedType}
          className={`ui-input ${error ? 'ui-input--error' : ''} ${hasToggle ? 'ui-input--has-toggle' : ''}`}
          aria-invalid={error ? 'true' : undefined}
          aria-describedby={describedBy}
          {...rest}
        />
        {hasToggle && (
          <button
            type="button"
            className="ui-input__toggle"
            onClick={() => setVisible((v) => !v)}
            aria-label={visible ? 'Hide password' : 'Show password'}
            tabIndex={0}
          >
            {visible ? <IconEyeOff size={18} /> : <IconEye size={18} />}
          </button>
        )}
      </div>
      {hint && !error && (
        <span className="ui-hint" id={`${inputId}-hint`}>
          {hint}
        </span>
      )}
      {error && (
        <span className="ui-error-text" id={`${inputId}-error`} role="alert">
          <IconAlertCircle size={14} />
          {error}
        </span>
      )}
    </div>
  );
};

export default Input;
