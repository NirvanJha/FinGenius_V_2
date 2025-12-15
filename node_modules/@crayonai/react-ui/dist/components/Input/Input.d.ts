import React from "react";
export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> {
    styles?: React.CSSProperties;
    className?: string;
    size?: "small" | "medium" | "large";
    hasError?: boolean;
}
export declare const Input: React.ForwardRefExoticComponent<InputProps & React.RefAttributes<HTMLInputElement>>;
//# sourceMappingURL=Input.d.ts.map