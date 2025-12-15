interface YAxisTickProps {
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
    orientation?: "left" | "right";
    tickFormatter?: (value: any) => string;
    index?: number;
    visibleTicksCount?: number;
    setLabelWidth: (label: string) => void;
}
declare const YAxisTick: React.FC<YAxisTickProps>;
export { YAxisTick };
export type { YAxisTickProps };
//# sourceMappingURL=YAxisTick.d.ts.map