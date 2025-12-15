import * as LabelPrimitive from "@radix-ui/react-label";
import React from "react";
export interface LabelProps extends React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root> {
    children: React.ReactNode;
    className?: string;
    style?: React.CSSProperties;
    disabled?: boolean;
    required?: boolean;
}
declare const Label: React.ForwardRefExoticComponent<LabelProps & React.RefAttributes<HTMLLabelElement>>;
export { Label };
//# sourceMappingURL=Label.d.ts.map