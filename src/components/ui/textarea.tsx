// components/ui/textarea.tsx
import React from 'react';

export const Textarea = ({ ...props }) => {
  return (
    <textarea
      className="w-full p-3 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
      {...props}
    />
  );
};
