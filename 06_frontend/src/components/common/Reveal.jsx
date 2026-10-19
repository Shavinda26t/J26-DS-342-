import React from 'react';
import useReveal from '../../hooks/useReveal';

const Reveal = ({ children, className = '', ...rest }) => {
  const ref = useReveal();
  return (
    <div ref={ref} className={`ui-reveal ${className}`.trim()} {...rest}>
      {children}
    </div>
  );
};

export default Reveal;
