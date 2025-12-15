export interface CarouselProps extends React.HTMLAttributes<HTMLDivElement> {
    itemsToScroll?: number;
    noSnap?: boolean;
    showButtons?: boolean;
    variant?: "card" | "sunk";
    onScrollLeftEnabled?: (enabled: boolean) => void;
    onScrollRightEnabled?: (enabled: boolean) => void;
}
export interface CarouselRef {
    scroll: (direction: "left" | "right") => void;
    scrollDivRef: React.RefObject<HTMLDivElement | null>;
}
export declare const Carousel: import("react").ForwardRefExoticComponent<CarouselProps & import("react").RefAttributes<CarouselRef>>;
export declare const CarouselContent: import("react").ForwardRefExoticComponent<import("react").HTMLAttributes<HTMLDivElement> & import("react").RefAttributes<HTMLDivElement>>;
export declare const CarouselItem: import("react").ForwardRefExoticComponent<import("react").HTMLAttributes<HTMLDivElement> & import("react").RefAttributes<HTMLDivElement>>;
export declare const CarouselPrevious: import("react").ForwardRefExoticComponent<Omit<import("../IconButton").IconButtonProps & import("react").RefAttributes<HTMLButtonElement>, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
export declare const CarouselNext: import("react").ForwardRefExoticComponent<Omit<import("../IconButton").IconButtonProps & import("react").RefAttributes<HTMLButtonElement>, "ref"> & import("react").RefAttributes<HTMLButtonElement>>;
//# sourceMappingURL=Carousel.d.ts.map