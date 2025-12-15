import React from "react";
export interface TextAreaProps extends Omit<React.TextareaHTMLAttributes<HTMLTextAreaElement>, "size"> {
    className?: string;
    placeholder?: string;
    rows?: number;
    hasError?: boolean;
}
declare const TextArea: React.ForwardRefExoticComponent<TextAreaProps & React.RefAttributes<HTMLTextAreaElement>>;
export { TextArea };
//# sourceMappingURL=TextArea.d.ts.map