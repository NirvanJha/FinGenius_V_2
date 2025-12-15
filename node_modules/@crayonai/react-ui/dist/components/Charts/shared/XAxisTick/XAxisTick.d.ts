import React from "react";
import { XAxisTickVariant } from "../../types";
interface XAxisTickProps {
    x?: number;
    y?: number;
    payload?: {
        value: any;
        coordinate?: number;
        tickCoord?: number;
        index?: number;
        offset?: number;
        isShow?: boolean;
    };
    textAnchor?: "start" | "middle" | "end";
    verticalAnchor?: "start" | "middle" | "end";
    fill?: string;
    stroke?: string;
    width?: number;
    height?: number;
    className?: string;
    orientation?: "top" | "bottom";
    tickFormatter?: (value: any) => string;
    index?: number;
    visibleTicksCount?: number;
    variant?: XAxisTickVariant;
    widthOfGroup?: number;
    labelHeight?: number;
    onMouseEnter?: (tickProps: XAxisTickProps) => void;
    onMouseLeave?: (tickProps: XAxisTickProps) => void;
}
declare const XAxisTick: React.ForwardRefExoticComponent<XAxisTickProps & React.RefAttributes<SVGGElement>>;
export { XAxisTick };
export type { XAxisTickProps };
//# sourceMappingURL=XAxisTick.d.ts.map