import React, { useEffect } from 'react';
import { IconX } from './Icons';

const Modal = ({ open, onClose, title, size = 'md', footer = null, children }) => {
  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event) => {
      if (event.key === 'Escape' && onClose) onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="ui-modal__overlay"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && onClose) onClose();
      }}
    >
      <div
        className={`ui-modal ${size === 'lg' ? 'ui-modal--lg' : ''}`.trim()}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === 'string' ? title : 'Dialog'}
      >
        <div className="ui-modal__header">
          <h3 className="ui-modal__title">{title}</h3>
          <button type="button" className="ui-icon-btn" onClick={onClose} aria-label="Close dialog">
            <IconX size={18} />
          </button>
        </div>
        <div className="ui-modal__body">{children}</div>
        {footer && <div className="ui-modal__footer">{footer}</div>}
      </div>
    </div>
  );
};

export default Modal;
