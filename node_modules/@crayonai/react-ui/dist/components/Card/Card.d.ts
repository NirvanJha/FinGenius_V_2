import React from "react";
type CardVariant = "clear" | "card" | "sunk";
type CardWidth = "standard" | "full";
export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
    variant?: CardVariant;
    width?: CardWidth;
}
export declare const Card: React.ForwardRefExoticComponent<CardProps & React.RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=Card.d.ts.map