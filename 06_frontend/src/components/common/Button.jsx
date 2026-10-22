import React from 'react';

const Button = ({
  variant = 'primary',
  size = 'md',
  type = 'button',
  fullWidth = false,
  loading = false,
  disabled = false,
  leftIcon = null,
  rightIcon = null,
  className = '',
  children,
  ...rest
}) => {
  const classes = [
    'ui-btn',
    `ui-btn--${variant}`,
    size !== 'md' && `ui-btn--${size}`,
    fullWidth && 'ui-btn--block',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <button type={type} className={classes} disabled={disabled || loading} {...rest}>
      {loading ? <span className="ui-spinner ui-spinner--sm" aria-hidden="true" /> : leftIcon}
      <span>{children}</span>
      {!loading && rightIcon}
    </button>
  );
};

export default Button;
