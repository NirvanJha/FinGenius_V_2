import React from "react";
export type SVGXAxisTickVariant = "singleLine" | "angled";
interface SVGXAxisTickProps {
    x?: number;
    y?: number;
    payload?: {
        value: any;
    };
    className?: string;
    angle?: number;
    textAnchor?: "start" | "middle" | "end";
    tickFormatter?: (value: any) => string;
    dy?: number;
}
declare const SVGXAxisTick: React.ForwardRefExoticComponent<SVGXAxisTickProps & React.RefAttributes<SVGTextElement>>;
export { SVGXAxisTick };
export type { SVGXAxisTickProps };
//# sourceMappingURL=SVGXAxisTick.d.ts.map