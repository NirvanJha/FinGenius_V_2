import React from "react";
export interface HintProps extends React.ComponentPropsWithoutRef<"div"> {
    children: React.ReactNode;
    className?: string;
    style?: React.CSSProperties;
    hasError?: boolean;
}
declare const Hint: React.ForwardRefExoticComponent<HintProps & React.RefAttributes<HTMLDivElement>>;
export { Hint };
//# sourceMappingURL=Hint.d.ts.map