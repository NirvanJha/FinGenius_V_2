import { FunctionComponent } from "react";
interface LineInBarShapeProps {
    x?: number;
    y?: number;
    width?: number;
    height?: number;
    fill?: string;
    stroke?: string;
    strokeWidth?: number | string;
    radius?: number | number[];
    internalLineColor?: string;
    internalLineWidth?: number;
    isHovered?: boolean;
    hoveredCategory?: string | number | null;
    categoryKey?: string;
    payload?: any;
    variant?: "grouped" | "stacked";
    stackGap?: number;
    orientation?: "vertical" | "horizontal";
    hasNegativeValueInStack?: boolean;
    [key: string]: any;
}
declare const LineInBarShape: FunctionComponent<LineInBarShapeProps>;
export { LineInBarShape };
//# sourceMappingURL=LineInBarShape.d.ts.map