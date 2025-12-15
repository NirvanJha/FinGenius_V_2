import React from "react";
type TextCalloutVariant = "neutral" | "info" | "warning" | "success" | "danger";
export interface TextCalloutProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "title"> {
    variant?: TextCalloutVariant;
    title?: React.ReactNode;
    description?: React.ReactNode;
}
export declare const TextCallout: React.ForwardRefExoticComponent<TextCalloutProps & React.RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=TextCallout.d.ts.map