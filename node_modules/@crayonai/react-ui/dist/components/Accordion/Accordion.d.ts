import * as AccordionPrimitive from "@radix-ui/react-accordion";
import React from "react";
type AccordionVariant = "clear" | "card" | "sunk";
export type AccordionProps = (AccordionPrimitive.AccordionSingleProps | AccordionPrimitive.AccordionMultipleProps) & {
    variant?: AccordionVariant;
};
export declare const Accordion: React.ForwardRefExoticComponent<AccordionProps & React.RefAttributes<HTMLDivElement>>;
export interface AccordionItemProps extends React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Item> {
    className?: string;
    style?: React.CSSProperties;
    value: string;
}
export declare const AccordionItem: React.ForwardRefExoticComponent<AccordionItemProps & React.RefAttributes<HTMLDivElement>>;
export interface AccordionTriggerProps extends React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Trigger> {
    className?: string;
    style?: React.CSSProperties;
    icon?: React.ReactNode;
    text: React.ReactNode;
}
export declare const AccordionTrigger: React.ForwardRefExoticComponent<AccordionTriggerProps & React.RefAttributes<HTMLButtonElement>>;
export interface AccordionContentProps extends React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content> {
    className?: string;
    style?: React.CSSProperties;
    children?: React.ReactNode;
}
export declare const AccordionContent: React.ForwardRefExoticComponent<AccordionContentProps & React.RefAttributes<HTMLDivElement>>;
export {};
//# sourceMappingURL=Accordion.d.ts.map