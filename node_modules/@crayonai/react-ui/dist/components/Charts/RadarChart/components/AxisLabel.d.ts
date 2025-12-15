import React from "react";
interface AxisLabelProps {
    x?: number;
    y?: number;
    textAnchor?: string;
    payload?: {
        value: string;
    };
    className?: string;
    portalContainerRef?: React.RefObject<HTMLDivElement | null>;
    [key: string]: any;
}
export declare const AxisLabel: React.FC<AxisLabelProps>;
export {};
//# sourceMappingURL=AxisLabel.d.ts.map