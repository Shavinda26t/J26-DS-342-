import React from 'react';

const LoadingSpinner = ({ size = 'md', label = null }) => (
  <div className="ui-loading" role="status" aria-live="polite">
    <span className={`ui-spinner ${size !== 'md' ? `ui-spinner--${size}` : ''}`.trim()} aria-hidden="true" />
    {label && <span className="ui-loading__label">{label}</span>}
  </div>
);

export default LoadingSpinner;
